import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from datetime import datetime, timezone
from functools import reduce
from inspect import isawaitable
from itertools import chain, islice
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias, TypeVar

from pydantic import Field, TypeAdapter, ValidationError

from .models import (
    Activity,
    Claim,
    Coverage,
    Evidence,
    Execution,
    ExecutionContent,
    FindingDraft,
    InFlight,
    ModelMessage,
    ModelRequest,
    ModelResult,
    Record,
    Result,
    Review,
    ReviewSpan,
    ReviewVerdict,
    RunAssessment,
    Sample,
    ToolCount,
    TracePart,
)
from .prompts import PROMPTS
from .trace_store import TraceStore, overview_content, trace_store


class Observation(Record):
    check_id: str
    kind: Literal["issue", "pattern"] = "issue"
    summary: str
    evidence: tuple[Evidence, ...] = Field(default=())


class Extraction(Record):
    observations: tuple[Observation, ...] = ()
    cannot_assess: bool = False
    reasoning: str = Field(default="", max_length=800)


class SpanRead(Record):
    span_id: str
    offset: int = Field(default=0, ge=0)


class TraceReview(Extraction):
    feedback_page: int | None = Field(default=None, ge=0)
    reads: tuple[SpanRead, ...] = Field(default=())


class Candidate(Record):
    check_id: str
    kind: Literal["issue", "pattern"] = "issue"
    title: str
    hypothesis: str
    execution_ids: tuple[str, ...]
    existing_finding_id: str | None = None


class Clusters(Record):
    candidates: tuple[Candidate, ...] = ()


class Decision(Record):
    action: Literal["read", "evidence", "observations", "catalog", "feedback", "submit", "inconclusive"]
    page: int = Field(default=0, ge=0)
    execution_id: str | None = None
    cursor: str = ""
    offset: int = Field(default=0, ge=0)
    finding: FindingDraft | None = None


class FinalDecision(Record):
    action: Literal["submit", "inconclusive"]
    finding: FindingDraft | None = None


class Examined(Record):
    execution: Execution
    observations: tuple[Observation, ...]
    parts: tuple[TracePart, ...]
    partial: bool
    cannot_assess: bool
    error: str = ""
    reasoning: str = ""
    shown: tuple[TracePart, ...] = ()
    tool_calls: tuple[ToolCount, ...] = ()


class Investigation(Record):
    finding: FindingDraft | None
    parts: tuple[TracePart, ...]
    error: str = ""


ModelCall: TypeAlias = Callable[[ModelRequest], Awaitable[ModelResult]]
ReadContent: TypeAlias = Callable[[str, str, int], Awaitable[ExecutionContent]]


class ReportProgress(Protocol):
    def __call__(
        self,
        stage: str | None,
        coverage: Coverage | None,
        review: Review | None = None,
        reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> Awaitable[None]: ...


ResponseT = TypeVar("ResponseT", bound=Record)


class ValidationIssue(Record):
    type: str
    loc: tuple[str | int, ...]
    msg: str


def validation_details(error: ValidationError) -> str:
    issues: Final = TypeAdapter(tuple[ValidationIssue, ...]).validate_json(
        error.json(include_input=False, include_context=False, include_url=False)
    )
    return "\n".join(
        f"{'.'.join(str(part) for part in issue.loc) or '$'}: {issue.msg} [{issue.type}]"
        if issue.type != "extra_forbidden"
        else "Unexpected field: Extra inputs are not permitted [extra_forbidden]"
        for issue in issues
    )


class AnalysisResponseError(ValueError):
    pass


class AnalysisContextExceeded(AnalysisResponseError):
    def __init__(self, request: ModelRequest) -> None:
        self.request: Final = request
        super().__init__("The analysis conversation exceeds the model's context window.")


async def structured_response(
    request: ModelRequest,
    schema: type[ResponseT],
    model: ModelCall,
    validate: Callable[[ResponseT], str | None | Awaitable[str | None]] = lambda _: None,
) -> ResponseT:
    parsed, _ = await structured_response_with_history(request, schema, model, validate)
    return parsed


async def structured_response_with_history(
    request: ModelRequest,
    schema: type[ResponseT],
    model: ModelCall,
    validate: Callable[[ResponseT], str | None | Awaitable[str | None]] = lambda _: None,
) -> tuple[ResponseT, tuple[ModelMessage, ...]]:
    response: Final = await model(request)
    if response.context_exceeded:
        raise AnalysisContextExceeded(request)
    parsed, problem = await checked_response(response, schema, validate)
    if parsed is not None:
        return parsed, (*request.messages, ModelMessage(role="assistant", content=response.content))
    correction: Final = (
        "\nYour previous response did not match the required response contract. Generate a new response "
        "from the original evidence, correcting these validation errors: " + problem
    )
    repair: Final = request.model_copy(
        update=MappingProxyType(
            {
                "messages": (
                    *request.messages,
                    ModelMessage(role="assistant", content=response.content),
                    ModelMessage(role="user", content=correction),
                )
            }
            if request.messages
            else {"prompt": request.prompt + correction}
        )
    )
    repaired: Final = await model(repair)
    if repaired.context_exceeded:
        raise AnalysisContextExceeded(repair)
    corrected, detail = await checked_response(repaired, schema, validate)
    if corrected is not None:
        return corrected, (*repair.messages, ModelMessage(role="assistant", content=repaired.content))
    stage: Final = MappingProxyType(
        {
            "extract": "Reading executions",
            "cluster": "Grouping observations",
            "investigate": "Checking original evidence",
        }
    )[request.purpose]
    stopped: Final = (
        " Model output was truncated (finish_reason=length)."
        if repaired.finish_reason == "length"
        else " Model output was blocked (finish_reason=content_filter)."
        if repaired.finish_reason == "content_filter"
        else ""
    )
    raise AnalysisResponseError(
        f"{stage} failed: {schema.__name__} response invalid after 2 attempts.{stopped}\n{detail}"
    )


async def checked_response(
    response: ModelResult,
    schema: type[ResponseT],
    validate: Callable[[ResponseT], str | None | Awaitable[str | None]],
) -> tuple[ResponseT | None, str]:
    try:
        parsed: Final = schema.model_validate_json(response.content)
        if response.finish_reason:
            return None, f"Model did not finish its response (finish_reason={response.finish_reason})"
    except ValueError as error:
        return None, validation_details(error) if isinstance(error, ValidationError) else str(error)
    validation: Final = validate(parsed)
    invalid: Final = await validation if isawaitable(validation) else validation
    return (None, invalid) if invalid else (parsed, "")


def evidence_valid(evidence: Evidence, parts: tuple[TracePart, ...]) -> bool:
    return any(
        p.execution_id == evidence.execution_id
        and p.span_id == evidence.span_id
        and any(evidence.quote in segment for segment in p.content.split("\n[... content omitted ...]\n"))
        for p in parts
    )


BatchItem = TypeVar("BatchItem")
BatchResult = TypeVar("BatchResult")
ANALYSIS_CONCURRENCY: Final = 8


async def concurrent_results(
    items: tuple[BatchItem, ...],
    operation: Callable[[BatchItem], Awaitable[BatchResult]],
    concurrency: int = ANALYSIS_CONCURRENCY,
) -> AsyncGenerator[BatchResult, None]:
    async def operate(item: BatchItem) -> BatchResult:
        return await operation(item)

    remaining: Final = iter(enumerate(items))
    pending = frozenset(  # rebind-ok: replace the bounded set as tasks finish
        asyncio.create_task(operate(item)) for _, item in islice(remaining, concurrency)
    )
    try:
        while pending:
            done, waiting = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            pending = frozenset((*waiting, *done))
            for task in done:
                yield await task
                pending = pending - frozenset((task,))
                for _, item in islice(remaining, 1):
                    pending = pending | frozenset((asyncio.create_task(operate(item)),))
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def partition_items(
    items: tuple[BatchItem, ...], size: Callable[[BatchItem], int], limit: int
) -> tuple[tuple[BatchItem, ...], ...]:
    def append_item(batches: tuple[tuple[BatchItem, ...], ...], item: BatchItem) -> tuple[tuple[BatchItem, ...], ...]:
        if not batches or sum(size(value) for value in batches[-1]) + size(item) > limit:
            return (*batches, (item,))
        return (*batches[:-1], (*batches[-1], item))

    return reduce(append_item, items, ())


def partition_content(parts: tuple[TracePart, ...], limit: int = 24000) -> tuple[tuple[TracePart, ...], ...]:
    return partition_items(parts, lambda part: len(part.model_dump_json()) + 20, limit)


async def read_execution(execution: Execution, read: ReadContent, store: TraceStore) -> ExecutionContent:
    cursor = ""  # rebind-ok: advance a database cursor until exhaustion
    partial = False  # rebind-ok: preserve incomplete source status across pages
    while True:
        page = await read(execution.id, cursor, 0)
        store.add(page.parts)
        partial = partial or page.partial
        if not page.next_cursor or page.next_cursor == cursor:
            return page.model_copy(update=MappingProxyType({"parts": (), "partial": partial}))
        cursor = page.next_cursor


async def extract(claim: Claim, execution: Execution, read: ReadContent, model: ModelCall) -> Examined:
    with trace_store() as store:
        try:
            return await extract_stored(claim, execution, read, model, store)
        except (ValidationError, AnalysisResponseError) as error:
            return Examined(
                execution=execution,
                observations=(),
                parts=(),
                partial=True,
                cannot_assess=True,
                error=validation_details(error) if isinstance(error, ValidationError) else str(error),
            )


async def extract_stored(
    claim: Claim, execution: Execution, read: ReadContent, model: ModelCall, store: TraceStore
) -> Examined:
    page: Final = await read_execution(execution, read, store)
    root_count: Final = sum(not p.parent_span_id for p in store.parts())
    first_root: Final = next((p for p in store.parts() if not p.parent_span_id), None)
    span_count: Final = store.count()
    feedback: Final = feedback_pages(claim)

    async def fetch(request: SpanRead) -> tuple[TracePart, ...]:
        previous: Final = store.previous(request.span_id)
        content: Final = await read(execution.id, previous, request.offset)
        return tuple(p for p in content.parts if p.span_id == request.span_id)

    async def examine(catalog: tuple[tuple[str, str, str, str, str], ...]) -> Examined:
        feedback_page = 0  # rebind-ok: navigate bounded feedback pages
        feedback_seen: set[int] = {0}  # mutable-ok: detect feedback navigation loops
        must_decide = False  # rebind-ok: unavailable evidence requires a final decision
        previous = TraceReview()  # rebind-ok: model state advances after evidence reads
        reads: tuple[SpanRead, ...] = ()  # rebind-ok: retain completed reads to detect loops
        additional: tuple[TracePart, ...] = ()  # rebind-ok: retain evidence fetched during this review

        async def review(
            previous: TraceReview,
            reads: tuple[SpanRead, ...],
            additional: tuple[TracePart, ...],
            feedback_page: int,
            must_decide: bool,
        ) -> TraceReview:
            prompt: Final = json.dumps(
                {
                    "task": PROMPTS.review,
                    "navigation": "The current feedback page is already included. Only request a different feedback_page "
                    "when feedback_pages>1. Zero feedback_pages means there is no feedback to consult. "
                    "When must_decide=true, return final observations without further reads or navigation.",
                    "must_decide": must_decide,
                    "context": claim.job.settings.context,
                    "checks": tuple(c.model_dump() for c in claim.job.settings.analysis_checks),
                    "execution": execution.model_dump(),
                    "catalog_complete": page.next_cursor is None and len(catalog) == span_count,
                    "catalog_fields": ("span_id", "parent_span_id", "name", "kind", "preview"),
                    "catalog": catalog,
                    "task_and_outcome": tuple(
                        p.model_copy(update=MappingProxyType({"content": overview_content(p, root_count)})).model_dump()
                        for p in (first_root,)
                        if p is not None
                    ),
                    "read_evidence": tuple(p.model_dump() for p in additional),
                    "previous_observations": tuple(o.model_dump() for o in previous.observations),
                    "completed_read_count": len(reads),
                    "last_completed_read": reads[-1].model_dump() if reads else None,
                    "feedback": feedback[feedback_page] if feedback else (),
                    "feedback_page": feedback_page,
                    "feedback_pages": len(feedback),
                    "response_schema": Extraction.model_json_schema()
                    if must_decide
                    else TraceReview.model_json_schema(),
                },
                ensure_ascii=False,
            )
            request: Final = ModelRequest(purpose="extract", prompt=prompt)
            if must_decide:
                final: Final = await structured_response(request, Extraction, model)
                return TraceReview(
                    observations=final.observations, cannot_assess=final.cannot_assess, reasoning=final.reasoning
                )
            return await structured_response(request, TraceReview, model)

        response: TraceReview
        requested: tuple[SpanRead, ...]
        fetched: tuple[tuple[TracePart, ...], ...]
        while True:
            response = await review(previous, reads, additional, feedback_page, must_decide)
            if must_decide or (not response.reads and response.feedback_page in (None, feedback_page)):
                break
            if response.feedback_page is not None and response.feedback_page != feedback_page:
                if response.feedback_page >= len(feedback) or response.feedback_page in feedback_seen:
                    must_decide = True
                else:
                    feedback_page = response.feedback_page
                    feedback_seen.add(feedback_page)
                previous = response
                continue
            requested = tuple(r for r in response.reads if r not in reads and store.get(r.span_id) is not None)
            if not requested:
                must_decide = True
                previous = response
                continue
            fetched = tuple([parts async for parts in concurrent_results(requested, fetch)])
            if not any(p.content for p in chain.from_iterable(fetched)):
                must_decide = True
                previous = response
                continue
            previous = response
            reads = (*reads, *requested)
            store.add_reads(tuple(chain.from_iterable(fetched)))
            additional = tuple(chain.from_iterable(fetched))
        cited_evidence: Final = tuple(chain.from_iterable(o.evidence for o in response.observations))
        verified: Final = tuple(store.evidence(e) for e in cited_evidence)
        evidence: Final = tuple(dict.fromkeys(p for p in verified if p is not None))
        observations: Final = tuple(
            o
            for o in response.observations
            if o.check_id in frozenset(c.id for c in claim.job.settings.analysis_checks)
            and o.evidence
            and all(evidence_valid(e, evidence) for e in o.evidence)
        )
        invalid_observations: Final = len(observations) != len(response.observations)
        return Examined(
            execution=execution,
            observations=observations,
            parts=evidence,
            partial=page.partial or page.next_cursor is not None or bool(response.reads) or invalid_observations,
            cannot_assess=not span_count or response.cannot_assess or bool(response.reads) or invalid_observations,
            reasoning=response.reasoning,
        )

    reviews: Final = tuple([await examine(catalog) for catalog in store.catalogs(root_count)])
    observations: Final = tuple(chain.from_iterable(item.observations for item in reviews))
    cited: Final = frozenset(e.span_id for e in chain.from_iterable(o.evidence for o in observations))
    retained: Final = tuple(
        p for p in chain.from_iterable(r.parts for r in reviews) if p.span_id in cited or not p.parent_span_id
    )
    leading: Final = MappingProxyType(
        {
            p.span_id: p
            for p in (*((first_root,) if first_root else ()), *(p for p in store.parts() if p.span_id in cited))
        }
    )
    shown: Final = islice(chain(leading.values(), (p for p in store.parts() if p.span_id not in leading)), 8)
    return Examined(
        execution=execution,
        observations=observations,
        parts=tuple(dict.fromkeys((*retained, *((first_root,) if first_root else ())))),
        partial=any(r.partial for r in reviews),
        cannot_assess=not reviews or all(r.cannot_assess for r in reviews),
        reasoning=" ".join(r.reasoning for r in reviews if r.reasoning),
        shown=tuple(p.model_copy(update=MappingProxyType({"content": overview_content(p, root_count)})) for p in shown),
    )


def review_of(examined: Examined, model: str, duration_ms: int, at: datetime) -> Review:
    execution: Final = examined.execution
    cited: Final = frozenset(
        (e.execution_id, e.span_id) for e in chain.from_iterable(o.evidence for o in examined.observations)
    )
    return Review(
        execution_id=execution.id,
        trace_id=execution.trace_id,
        agent=execution.service or execution.name,
        name=execution.name,
        spans=tuple(
            ReviewSpan(
                span_id=p.span_id,
                name=p.name[:120],
                kind=p.kind[:40],
                preview=p.content[:240],
                cited=(p.execution_id, p.span_id) in cited,
            )
            for p in examined.shown[:8]
        ),
        reasoning=examined.reasoning[:800],
        verdicts=tuple(
            ReviewVerdict(check_id=o.check_id, kind=o.kind, summary=o.summary[:300])
            for o in examined.observations
            if any(quote.execution_id == execution.id and quote.role == "support" for quote in o.evidence)
        ),
        cannot_assess=examined.cannot_assess,
        model=model,
        duration_ms=max(duration_ms, 0),
        at=at,
        tool_calls=examined.tool_calls,
    )


def feedback_pages(claim: Claim, check_id: str | None = None) -> tuple[tuple[tuple[str, str, str, str, str], ...], ...]:
    entries: Final = tuple(
        (f.id, f.check_id, f.title, f.status, f.reason)
        for f in claim.findings
        if check_id is None or f.check_id == check_id
    )
    return partition_items(entries, lambda row: len(json.dumps(row)), 8000)


async def investigate(
    claim: Claim,
    candidate: Candidate,
    examined: tuple[Examined, ...],
    read: ReadContent,
    model: ModelCall,
) -> Investigation:
    with trace_store() as store:
        try:
            return await investigate_stored(claim, candidate, examined, read, model, store)
        except (ValidationError, AnalysisResponseError) as error:
            return Investigation(
                finding=None,
                parts=(),
                error=validation_details(error) if isinstance(error, ValidationError) else str(error),
            )


async def investigate_stored(
    claim: Claim,
    candidate: Candidate,
    examined: tuple[Examined, ...],
    read: ReadContent,
    model: ModelCall,
    store: TraceStore,
) -> Investigation:
    additional: tuple[TracePart, ...] = ()  # rebind-ok: investigation accumulates fetched evidence
    navigation: ExecutionContent | None = None  # rebind-ok: last fetched page
    reads: tuple[Decision, ...] = ()  # rebind-ok: track completed tool requests to detect loops
    observation_page = 0  # rebind-ok: model controls navigation through observations
    evidence_page = 0  # rebind-ok: navigate all content in the fetched evidence batch
    evidence_seen = frozenset((0,))  # rebind-ok: reset navigation history when evidence changes
    catalog_page = 0  # rebind-ok: model controls navigation through the run catalog
    feedback_page = 0  # rebind-ok: navigate bounded prior finding pages
    feedback: Final = feedback_pages(claim, candidate.check_id)
    stalled = False  # rebind-ok: a repeated request requires a decision rather than a loop

    async def decide(
        additional: tuple[TracePart, ...],
        navigation: ExecutionContent | None,
        reads: tuple[Decision, ...],
        observation_page: int,
        evidence_page: int,
        catalog_page: int,
        feedback_page: int,
        stalled: bool,
    ) -> Decision | Investigation:
        relevant: Final = tuple(item for item in examined if item.execution.id in candidate.execution_ids)
        observations: Final = tuple(
            o
            for o in chain.from_iterable(item.observations for item in relevant)
            if o.check_id == candidate.check_id and o.kind == candidate.kind
        )
        supporting_batches: Final = partition_items(observations, lambda o: len(o.model_dump_json()), 16000)
        supporting: Final = supporting_batches[observation_page] if observation_page < len(supporting_batches) else ()
        cited: Final = frozenset(
            (e.execution_id, e.span_id) for e in chain.from_iterable(o.evidence for o in supporting)
        )
        selected: Final = tuple(chain.from_iterable(item.parts for item in relevant))
        unique: Final = MappingProxyType({(p.execution_id, p.span_id, p.content): p for p in (*selected, *additional)})
        recent: Final = navigation.parts if navigation else ()
        prioritized: Final = tuple(
            sorted(
                unique.values(),
                key=lambda p: (
                    p not in recent,
                    (p.execution_id, p.span_id) not in cited,
                    bool(p.parent_span_id),
                    p.kind == "llm",
                ),
            )
        )
        bounded: Final = partition_content(prioritized, 30000)
        evidence: Final = bounded[evidence_page] if evidence_page < len(bounded) else ()
        catalog_batches: Final = partition_items(
            (*relevant, *(item for item in examined if item not in relevant)),
            lambda item: len(item.execution.model_dump_json()),
            16000,
        )
        catalog: Final = catalog_batches[catalog_page] if catalog_page < len(catalog_batches) else ()
        prompt: Final = json.dumps(
            {
                "task": PROMPTS.investigate,
                "context": claim.job.settings.context,
                "questions": tuple(c.model_dump() for c in claim.job.settings.analysis_checks),
                "response_schema": Decision.model_json_schema() if not stalled else FinalDecision.model_json_schema(),
                "candidate": candidate.model_dump(exclude=MappingProxyType({"execution_ids": True})),
                "candidate_run_count": len(candidate.execution_ids),
                "supporting_observations": tuple(o.model_dump() for o in supporting),
                "total_supporting_observations": len(observations),
                "observation_page": observation_page,
                "observation_pages": len(supporting_batches),
                "catalog_page": catalog_page,
                "catalog_pages": len(catalog_batches),
                "workflow_outlines": tuple(
                    {
                        "execution_id": item.execution.id,
                        "recorded_span_count": item.execution.span_count,
                        "partial": item.partial,
                        "cannot_assess": item.cannot_assess,
                        "available_unique_spans": len(frozenset(p.span_id for p in item.parts)),
                        "span_names": tuple(sorted(frozenset(p.name for p in item.parts))),
                        "root_span_ids": tuple(p.span_id for p in item.parts if not p.parent_span_id),
                    }
                    for item in catalog
                ),
                "completed_read_count": len(reads),
                "last_completed_read": reads[-1].model_dump() if reads else None,
                "catalog": tuple(e.execution.model_dump() for e in catalog),
                "existing_findings_fields": ("id", "check_id", "title", "status", "reason"),
                "existing_findings": feedback[feedback_page] if feedback else (),
                "feedback_page": feedback_page,
                "feedback_pages": len(feedback),
                "evidence": tuple(p.model_dump() for p in evidence),
                "evidence_page": evidence_page,
                "evidence_pages": len(bounded),
                "must_decide": stalled,
                "last_read": navigation.model_dump(exclude=MappingProxyType({"parts": True})) if navigation else None,
            },
            ensure_ascii=False,
        )
        request: Final = ModelRequest(purpose="investigate", prompt=prompt)
        decision: Final = await investigation_decision(request, model, 1 if stalled else 2)
        if decision.action == "submit" and decision.finding:
            finding: Final = decision.finding
            known: Final = frozenset(c.id for c in claim.job.settings.analysis_checks)
            existing: Final = next((f for f in claim.findings if f.id == finding.existing_finding_id), None)
            valid_existing: Final = finding.existing_finding_id is None or (
                existing is not None and existing.check_id == finding.check_id
            )
            if (
                finding.check_id in known
                and finding.check_id == candidate.check_id
                and finding.kind == candidate.kind
                and any(e.role == "support" for e in finding.evidence)
                and valid_existing
                and all(
                    evidence_valid(e, tuple(unique.values())) or store.evidence(e) is not None for e in finding.evidence
                )
            ):
                return Investigation(finding=finding, parts=evidence)
        if stalled or decision.action not in ("read", "evidence", "observations", "catalog", "feedback"):
            return Investigation(finding=None, parts=evidence)
        page_count: Final = MappingProxyType(
            {
                "observations": len(supporting_batches),
                "evidence": len(bounded),
                "catalog": len(catalog_batches),
                "feedback": len(feedback),
            }
        )
        if decision.action in page_count and decision.page >= page_count[decision.action]:
            return Decision(action="inconclusive")
        return decision

    step_result: Decision | Investigation = (  # rebind-ok: next evidence turn changes the decision
        Decision(action="inconclusive")
    )
    while True:
        step_result = await decide(
            additional, navigation, reads, observation_page, evidence_page, catalog_page, feedback_page, stalled
        )
        if isinstance(step_result, Decision) and step_result.action == "inconclusive":
            stalled = True
            continue
        if isinstance(step_result, Investigation):
            return step_result
        if step_result.action == "evidence":
            if step_result.page in evidence_seen:
                stalled = True
            else:
                evidence_page = step_result.page
                evidence_seen = evidence_seen | frozenset((evidence_page,))
            continue
        if any(
            (r.action, r.execution_id, r.cursor, r.offset, r.page)
            == (step_result.action, step_result.execution_id, step_result.cursor, step_result.offset, step_result.page)
            for r in reads
        ):
            stalled = True
            continue
        reads = (*reads, step_result)
        if step_result.action == "observations":
            observation_page = step_result.page
            evidence_page = 0
            evidence_seen = frozenset((0,))
        elif step_result.action == "catalog":
            catalog_page = step_result.page
        elif step_result.action == "feedback":
            feedback_page = step_result.page
        elif any(e.execution.id == step_result.execution_id for e in examined):
            navigation = await read(step_result.execution_id or "", step_result.cursor, step_result.offset)
            if not any(p.content for p in navigation.parts):
                stalled = True
            store.add_reads(navigation.parts)
            additional = navigation.parts
            evidence_page = 0
            evidence_seen = frozenset((0,))
        else:
            return Investigation(finding=None, parts=additional)


async def investigation_decision(request: ModelRequest, model: ModelCall, steps: int) -> Decision:
    if steps > 1:
        return await structured_response(request, Decision, model)
    final: Final = await structured_response(request, FinalDecision, model)
    return Decision(action=final.action, finding=final.finding)


AnalyzeSample: TypeAlias = Callable[[Claim, Sample, ReadContent, ModelCall, ReportProgress], Awaitable[Result]]
ExtractExecution: TypeAlias = Callable[[Claim, Execution, ReadContent, ModelCall], Awaitable[Examined]]


async def analyze_sample(
    claim: Claim, sample: Sample, read: ReadContent, model: ModelCall, progress: ReportProgress
) -> Result:
    return await analyze_with(claim, sample, read, model, progress, analyze_executions)


async def analyze_with(
    claim: Claim,
    sample: Sample,
    read: ReadContent,
    model: ModelCall,
    progress: ReportProgress,
    analyze: AnalyzeSample,
) -> Result:
    originals: Final = MappingProxyType({f"r{index}": e for index, e in enumerate(sample.executions)})
    executions: Final = tuple(e.model_copy(update=MappingProxyType({"id": alias})) for alias, e in originals.items())

    async def read_alias(identity: str, cursor: str, offset: int) -> ExecutionContent:
        original: Final = originals[identity]
        page: Final = await read(original.id, cursor, offset)
        return page.model_copy(
            update=MappingProxyType(
                {
                    "execution": original.model_copy(update=MappingProxyType({"id": identity})),
                    "parts": tuple(
                        p.model_copy(update=MappingProxyType({"execution_id": identity})) for p in page.parts
                    ),
                }
            )
        )

    def original(identity: str) -> str:
        return originals[identity].id

    async def progress_original(
        stage: str | None,
        coverage: Coverage | None,
        review: Review | None = None,
        reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        await progress(
            stage,
            coverage,
            review and review.model_copy(update=MappingProxyType({"execution_id": original(review.execution_id)})),
            None
            if reading is None
            else tuple(
                r.model_copy(update=MappingProxyType({"execution_id": original(r.execution_id)})) for r in reading
            ),
            activity.model_copy(
                update=MappingProxyType(
                    {"execution_ids": tuple(original(identity) for identity in activity.execution_ids)}
                )
            )
            if activity is not None
            else None,
        )

    result: Final = await analyze(
        claim,
        sample.model_copy(update=MappingProxyType({"executions": executions})),
        read_alias,
        model,
        progress_original,
    )
    return result.model_copy(
        update=MappingProxyType(
            {
                "assessments": tuple(
                    a.model_copy(update=MappingProxyType({"execution_id": originals[a.execution_id].id}))
                    for a in result.assessments
                ),
                "findings": tuple(
                    f.model_copy(
                        update=MappingProxyType(
                            {
                                "evidence": tuple(
                                    e.model_copy(
                                        update=MappingProxyType({"execution_id": originals[e.execution_id].id})
                                    )
                                    for e in f.evidence
                                ),
                            }
                        )
                    )
                    for f in result.findings
                ),
            }
        )
    )


async def analyze_executions(
    claim: Claim,
    sample: Sample,
    read: ReadContent,
    model: ModelCall,
    progress: ReportProgress,
    *,
    extractor: ExtractExecution = extract,
) -> Result:
    base: Final = Coverage(eligible=sample.eligible, selected=len(sample.executions))
    if not sample.executions:
        return Result(coverage=base)
    slots: Final = asyncio.Semaphore(claim.job.settings.concurrency)

    async def limited_model(request: ModelRequest) -> ModelResult:
        async with slots:
            return await model(request)

    examined: Final = tuple(
        [item async for item in examine_executions(claim, sample, read, limited_model, progress, extractor=extractor)]
    )
    coverage: Final = base.model_copy(
        update=MappingProxyType(
            {
                "screened": len(examined),
                "partial": sum(e.partial for e in examined),
                "unassessable": sum(e.cannot_assess for e in examined),
            }
        )
    )
    assessments: Final = tuple(
        RunAssessment(
            execution_id=item.execution.id,
            issue_checks=tuple(sorted(frozenset(o.check_id for o in item.observations if o.kind == "issue"))),
            pattern_checks=tuple(sorted(frozenset(o.check_id for o in item.observations if o.kind == "pattern"))),
            cannot_assess=item.cannot_assess,
        )
        for item in examined
    )
    await progress("Grouping observations", coverage)
    observations: Final = tuple(chain.from_iterable(item.observations for item in examined))
    if not observations:
        return Result(
            coverage=coverage,
            assessments=assessments,
            error="\n\n".join(dict.fromkeys(item.error for item in examined if item.error)),
        )
    batches: Final = observation_batches(observations)
    grouping: Final = coverage.model_copy(update=MappingProxyType({"grouping_batches": len(batches)}))
    clusters: Final = await cluster_batches(batches, limited_model, progress, grouping)
    candidates: Final = clusters.candidates
    investigating: Final = grouping.model_copy(
        update=MappingProxyType({"grouped_batches": len(batches), "candidates": len(candidates)})
    )
    investigated: Final = tuple(
        [
            item
            async for item in investigate_candidates(
                claim, candidates, examined, read, limited_model, progress, investigating
            )
        ]
    )
    return Result(
        findings=tuple(item.finding for item in investigated if item.finding is not None),
        assessments=assessments,
        error="\n\n".join(dict.fromkeys(item.error for item in (*examined, *investigated) if item.error)),
        coverage=investigating.model_copy(
            update=MappingProxyType(
                {"investigated": len(candidates), "inconclusive": sum(item.finding is None for item in investigated)}
            )
        ),
    )


async def cluster_batches(
    batches: tuple[tuple[Observation, ...], ...],
    model: ModelCall,
    progress: ReportProgress,
    coverage: Coverage,
) -> Clusters:
    async def consolidate(batch: tuple[Observation, ...], previous: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
        incoming: Final = tuple(
            Candidate(
                check_id=o.check_id,
                kind=o.kind,
                title=o.summary,
                hypothesis=f"{o.kind}: {o.summary}",
                execution_ids=tuple(sorted(frozenset(e.execution_id for e in o.evidence))),
            )
            for o in batch
        )
        active = incoming  # rebind-ok: consolidate incoming patterns across registry pages
        retained: list[Candidate] = []  # mutable-ok: retain completed pages without copying the entire registry
        pages: Final = partition_items(previous, candidate_size, 16000)
        for prior in pages or ((),):
            continued, settled = await merge_candidates((*prior, *active), len(prior), model)
            active = continued
            retained.extend(settled)
        return (*retained, *active)

    candidates: tuple[Candidate, ...] = ()  # rebind-ok: fold observation batches into the pattern registry
    for index, batch in enumerate(batches):
        await progress(
            "Grouping observations", coverage.model_copy(update=MappingProxyType({"grouped_batches": index}))
        )
        candidates = await consolidate(batch, candidates)
    registry: tuple[Candidate, ...] = ()  # rebind-ok: compare every surviving candidate against all earlier patterns
    ordered: Final = tuple(sorted(candidates, key=lambda c: (c.check_id, c.kind)))
    for incoming in partition_items(ordered, candidate_size, 8000):
        kinds = frozenset((c.check_id, c.kind) for c in incoming)
        matching = tuple(c for c in registry if (c.check_id, c.kind) in kinds)
        unrelated = tuple(c for c in registry if (c.check_id, c.kind) not in kinds)
        carried = incoming
        retained: list[Candidate] = []  # mutable-ok: collect settled pages once
        for prior in partition_items(matching, candidate_size, 16000) or ((),):
            merged, settled = await merge_candidates((*prior, *carried), len(prior), model)
            carried = merged
            retained.extend(settled)
        registry = (*unrelated, *retained, *carried)
    return Clusters(candidates=registry)


def candidate_size(candidate: Candidate) -> int:
    return len(candidate.title) + len(candidate.hypothesis) + len(candidate.check_id) + 200


async def merge_candidates(
    candidates: tuple[Candidate, ...], prior_count: int, model: ModelCall
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...]]:
    identities: Final = MappingProxyType({f"p{i}": c for i, c in enumerate(candidates)})

    def validate_groups(groups: Clusters) -> str | None:
        references: Final = tuple(chain.from_iterable(c.execution_ids for c in groups.candidates))
        if len(references) != len(frozenset(references)):
            return "Each input reference must appear in exactly one group; do not duplicate it across findings."
        return None

    response: Final = await structured_response(
        ModelRequest(
            purpose="cluster",
            prompt=json.dumps(
                {
                    "task": PROMPTS.cluster,
                    "response_schema": Clusters.model_json_schema(),
                    "candidates": tuple(
                        c.model_copy(update=MappingProxyType({"execution_ids": (identity,)})).model_dump()
                        for identity, c in identities.items()
                    ),
                },
                ensure_ascii=False,
            ),
        ),
        Clusters,
        model,
        validate_groups,
    )
    valid: Final = tuple(
        c
        for c in response.candidates
        if c.execution_ids
        and all(
            identity in identities
            and identities[identity].check_id == c.check_id
            and identities[identity].kind == c.kind
            for identity in c.execution_ids
        )
    )
    used: Final = frozenset(chain.from_iterable(c.execution_ids for c in valid))
    expanded: Final = tuple(
        (
            c.model_copy(
                update=MappingProxyType(
                    {
                        "execution_ids": tuple(
                            sorted(
                                frozenset(
                                    chain.from_iterable(
                                        identities[identity].execution_ids for identity in c.execution_ids
                                    )
                                )
                            )
                        )
                    }
                )
            ),
            any(int(identity[1:]) >= prior_count for identity in c.execution_ids),
        )
        for c in valid
    )
    preserved: Final = (
        *expanded,
        *((c, int(identity[1:]) >= prior_count) for identity, c in identities.items() if identity not in used),
    )
    return tuple(c for c, active in preserved if active), tuple(c for c, active in preserved if not active)


async def examine_executions(
    claim: Claim,
    sample: Sample,
    read: ReadContent,
    model: ModelCall,
    progress: ReportProgress,
    *,
    extractor: ExtractExecution = extract,
) -> AsyncIterator[Examined]:
    reading: tuple[InFlight, ...] = ()  # rebind-ok: the in-flight set changes as each read starts and finishes
    screened = 0  # rebind-ok: counts finished reads for progress
    reporting: Final = asyncio.Lock()

    async def report(change: Callable[[tuple[InFlight, ...]], tuple[InFlight, ...]], review: Review | None) -> None:
        nonlocal reading
        async with reporting:
            reading = change(reading)
            coverage: Final = Coverage(eligible=sample.eligible, selected=len(sample.executions), screened=screened)
            await progress("Reading executions", coverage, review, reading)

    async def examine(execution: Execution) -> tuple[Examined, Review]:
        entry: Final = InFlight(
            execution_id=execution.id,
            trace_id=execution.trace_id,
            agent=execution.service or execution.name,
            started_at=datetime.now(timezone.utc),
        )
        await report(lambda current: (*current, entry), None)
        started: Final = time.perf_counter()
        examined: Final = await extractor(claim, execution, read, model)
        elapsed: Final = round((time.perf_counter() - started) * 1000)
        return examined, review_of(examined, claim.job.settings.model, elapsed, datetime.now(timezone.utc))

    await progress("Reading executions", Coverage(eligible=sample.eligible, selected=len(sample.executions)))
    async with aclosing(concurrent_results(sample.executions, examine, claim.job.settings.concurrency)) as results:
        async for item, review in results:
            screened += 1
            await report(
                lambda current, done=item.execution.id: tuple(r for r in current if r.execution_id != done), review
            )
            yield item


async def investigate_candidates(
    claim: Claim,
    candidates: tuple[Candidate, ...],
    examined: tuple[Examined, ...],
    read: ReadContent,
    model: ModelCall,
    progress: ReportProgress,
    coverage: Coverage,
) -> AsyncIterator[Investigation]:
    async def check(candidate: Candidate) -> Investigation:
        return await investigate(claim, candidate, examined, read, model)

    completed: Final = iter(range(1, len(candidates) + 1))
    inconclusive = 0  # rebind-ok: report unresolved candidates as each result arrives
    async with aclosing(concurrent_results(candidates, check, claim.job.settings.concurrency)) as results:
        async for investigation in results:
            inconclusive += int(investigation.finding is None)
            await progress(
                "Checking original evidence",
                coverage.model_copy(
                    update=MappingProxyType({"investigated": next(completed), "inconclusive": inconclusive})
                ),
            )
            yield investigation


def observation_batches(observations: tuple[Observation, ...]) -> tuple[tuple[Observation, ...], ...]:
    ordered: Final = tuple(sorted(observations, key=lambda observation: (observation.check_id, observation.kind)))
    return partition_items(ordered, lambda observation: len(observation.model_dump_json()), 16000)
