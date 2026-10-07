import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from .activity import ActivityTracker, observed_model, track_activity
from .agent_review import FINDINGS_TASK, Findings, review_context, validate_findings
from .agent_runtime import run_agent
from .agent_workspace import EvidenceReadError, EvidenceWorkspace, ReviewRecord, load_workspace
from .analysis import (
    AnalysisContextExceeded,
    AnalysisResponseError,
    AnalysisStopped,
    Candidate,
    Clusters,
    Examined,
    Extraction,
    ModelCall,
    Observation,
    ReadContent,
    ReportProgress,
    analyze_with,
    concurrent_results,
    examine_executions,
    merge_candidates,
    observation_batches,
)
from .models import (
    Activity,
    Claim,
    Coverage,
    Execution,
    FindingDraft,
    InFlight,
    ModelRequest,
    ModelResult,
    Record,
    Result,
    Review,
    ReviewVersion,
    RunAssessment,
    Sample,
)
from .reconciliation import reconcile_findings

ACCESS: Final[Literal["full", "tools", "python"]] = "python"


class CandidateInvestigation(Record):
    findings: tuple[FindingDraft, ...] = ()
    error: str = ""


class ReviewPlan(Record):
    execution_id: str
    content_version: str = ""
    previous: Review | None = None
    error: str = ""


async def plan_reviews(claim: Claim, workspace: EvidenceWorkspace) -> tuple[ReviewPlan, ...]:
    async def plan(execution: Execution) -> ReviewPlan:
        if claim.reviews is None:
            return ReviewPlan(execution_id=execution.id)
        try:
            version: Final = await workspace.fingerprint(execution.id)
        except EvidenceReadError as error:
            return ReviewPlan(execution_id=execution.id, error=str(error))
        previous: Final = next(
            (
                review
                for review in claim.reviews
                if review.execution_id == execution.id and review.content_version == version and review.extraction
            ),
            None,
        )
        return ReviewPlan(execution_id=execution.id, content_version=version, previous=previous)

    return tuple(
        [
            item
            async for item in concurrent_results(
                tuple(session.execution for session in workspace.sessions), plan, claim.job.settings.concurrency
            )
        ]
    )


async def analyze_sample(
    claim: Claim, sample: Sample, read: ReadContent, model: ModelCall, progress: ReportProgress
) -> Result:
    return await analyze_with(claim, sample, read, model, progress, analyze_context)


async def parallel_cluster_batches(
    batches: tuple[tuple[Observation, ...], ...],
    model: ModelCall,
    progress: ReportProgress,
    coverage: Coverage,
    concurrency: int,
) -> Clusters:
    async def group(item: tuple[int, tuple[Observation, ...]]) -> tuple[int, tuple[Candidate, ...]]:
        index, observations = item
        incoming: Final = tuple(
            Candidate(
                check_id=observation.check_id,
                kind=observation.kind,
                title=observation.summary,
                hypothesis=f"{observation.kind}: {observation.summary}",
                execution_ids=tuple(
                    sorted(frozenset(quote.execution_id for quote in observation.evidence if quote.role == "support"))
                ),
            )
            for observation in observations
        )
        async with track_activity(
            progress,
            identity=f"group:{index}",
            phase="group",
            label=f"Compare observation batch {index + 1}",
            execution_ids=tuple(
                sorted(frozenset(chain.from_iterable(candidate.execution_ids for candidate in incoming)))
            ),
        ) as activity:
            call: Final = observed_model(model, activity)
            try:
                merged, preserved = await merge_candidates(incoming, 0, call)
            except AnalysisContextExceeded:
                return index, await reconcile_registry(incoming, call)
            return index, (*preserved, *merged)

    completed: Final = iter(range(1, len(batches) + 1))
    grouped: tuple[tuple[int, tuple[Candidate, ...]], ...] = ()  # rebind-ok: retain completed independent batches
    async with aclosing(concurrent_results(tuple(enumerate(batches)), group, concurrency)) as results:
        async for result in results:
            grouped = (*grouped, result)
            await progress(
                "Grouping observations",
                coverage.model_copy(update=MappingProxyType({"grouped_batches": next(completed)})),
            )
    candidates: Final = tuple(chain.from_iterable(candidates for _, candidates in sorted(grouped)))
    if len(batches) < 2:
        return Clusters(candidates=candidates)
    return await reconcile_candidates(candidates, model, progress)


async def reconcile_candidates(
    candidates: tuple[Candidate, ...], model: ModelCall, progress: ReportProgress | None = None
) -> Clusters:
    ordered: Final = tuple(sorted(candidates, key=lambda candidate: (candidate.check_id, candidate.kind)))
    async with track_activity(
        progress,
        identity="reconcile",
        phase="reconcile",
        label="Compare candidate patterns",
        execution_ids=tuple(
            sorted(frozenset(chain.from_iterable(candidate.execution_ids for candidate in candidates)))
        ),
    ) as activity:
        call: Final = observed_model(model, activity)
        try:
            merged, preserved = await merge_candidates(ordered, 0, call)
        except AnalysisContextExceeded:
            return Clusters(candidates=await reconcile_registry(ordered, call))
        return Clusters(candidates=(*preserved, *merged))


async def reconcile_registry(candidates: tuple[Candidate, ...], model: ModelCall) -> tuple[Candidate, ...]:
    registry: tuple[Candidate, ...] = ()  # rebind-ok: compare each incoming cause against all retained groups
    for candidate in candidates:
        if not registry:
            registry = (candidate,)
            continue
        active, preserved = await merge_registry_page(registry, (candidate,), model)
        registry = (*preserved, *active)
    return registry


async def merge_registry_page(
    prior: tuple[Candidate, ...], active: tuple[Candidate, ...], model: ModelCall
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...]]:
    try:
        return await merge_candidates((*prior, *active), len(prior), model)
    except AnalysisContextExceeded as error:
        if len(prior) <= 1:
            raise AnalysisResponseError(
                "The smallest candidate comparison exceeds the analysis model's context window. "
                "Use a model with more context to compare these candidate patterns."
            ) from error
        midpoint: Final = len(prior) // 2
        continued, earlier = await merge_registry_page(prior[:midpoint], active, model)
        merged, later = await merge_registry_page(prior[midpoint:], continued, model)
        return merged, (*earlier, *later)


async def investigate_context_candidate(
    claim: Claim,
    candidate: Candidate,
    workspace: EvidenceWorkspace,
    model: ModelCall,
    *,
    access: Literal["full", "tools", "python"] = ACCESS,
    activity: ActivityTracker | None = None,
) -> CandidateInvestigation:
    try:
        response: Final = await run_agent(
            stage="context_investigation",
            task=FINDINGS_TASK
            + "\nInvestigate the supplied candidate against original evidence, including counterexamples. "
            "Reviewer records contain the initial observations and exact evidence references. Use read_reviews "
            "for the candidate's sessions and search_reviews to compare other sessions when useful. You can "
            "inspect every sampled session and its nested agents. Finalize findings about the supplied "
            "candidate's check and underlying cause or causes. Use unrelated successes as context or "
            "counterevidence rather than additional success findings; other candidates have their own "
            "investigators. Preserve distinct supported causes if the candidate conflates them. Return every "
            "supported finding for this assignment, or an empty findings list if the evidence does not support it.",
            purpose="investigate",
            claim=claim,
            workspace=workspace,
            model=model,
            schema=Findings,
            initial_evidence=(
                await workspace.get_parts(execution_ids=candidate.execution_ids) if access == "full" else ()
            ),
            supplied=candidate.model_dump_json(),
            validate=lambda findings: validate_findings(claim, workspace, findings),
            enable_python=access == "python",
            activity=activity,
        )
        return CandidateInvestigation(findings=response.findings)
    except (AnalysisResponseError, EvidenceReadError) as error:
        return CandidateInvestigation(error=str(error))


async def collect_reviews(reviews: AsyncGenerator[Examined, None]) -> tuple[tuple[Examined, ...], str]:
    completed: tuple[Examined, ...] = ()  # rebind-ok: retain completed reviews if a later model call stops
    try:
        async with aclosing(reviews):
            async for review in reviews:
                completed = (*completed, review)
    except AnalysisStopped as error:
        return completed, str(error)
    return completed, ""


async def analyze_context(
    claim: Claim,
    sample: Sample,
    read: ReadContent,
    model: ModelCall,
    progress: ReportProgress,
    *,
    access: Literal["full", "tools", "python"] = ACCESS,
) -> Result:
    base: Final = Coverage(eligible=sample.eligible, selected=len(sample.executions))
    if not sample.executions:
        return Result(coverage=base)
    async with track_activity(
        progress,
        identity="load",
        phase="load",
        label="Prepare evidence workspace",
        execution_ids=tuple(execution.id for execution in sample.executions),
    ):
        workspace: Final = await load_workspace(sample, read, claim.job.settings.concurrency)
    await progress("Checking for reusable reviews", base)
    plans: Final = MappingProxyType({plan.execution_id: plan for plan in await plan_reviews(claim, workspace)})
    reusable: Final = sum(plan.previous is not None for plan in plans.values())

    async def planned_progress(
        stage: str | None,
        coverage: Coverage | None,
        review: Review | None = None,
        reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        await progress(
            stage,
            coverage.model_copy(update=MappingProxyType({"reusable": reusable})) if coverage is not None else None,
            review,
            reading,
            activity,
        )

    await planned_progress("Reuse plan ready", base)
    slots: Final = asyncio.Semaphore(claim.job.settings.concurrency)

    async def limited(request: ModelRequest) -> ModelResult:
        async with slots:
            return await model(request)

    async def extract(claim: Claim, execution: Execution, _read: ReadContent, model: ModelCall) -> Examined:
        session: Final = next(session for session in workspace.sessions if session.execution.id == execution.id)
        async with track_activity(
            progress,
            identity=f"review:{execution.id}",
            phase="review",
            label=execution.service or execution.name,
            execution_ids=(execution.id,),
        ) as activity:
            try:
                plan: Final = plans[execution.id]
                if plan.error:
                    return Examined(
                        execution=execution,
                        observations=(),
                        parts=(),
                        partial=True,
                        cannot_assess=True,
                        error=plan.error,
                        reasoning=plan.error,
                    )
                version: Final = plan.content_version
                previous: Final = plan.previous
                if previous is not None and previous.extraction is not None:
                    return Examined(
                        execution=execution,
                        observations=previous.extraction.observations,
                        parts=(),
                        partial=previous.partial,
                        cannot_assess=previous.cannot_assess,
                        reasoning=previous.reasoning,
                        content_version=version,
                        reused=True,
                        consolidated=previous.consolidated,
                    )
                reviewed: Final = await review_context(
                    claim.model_copy(update=MappingProxyType({"findings": ()})) if claim.reviews is not None else claim,
                    session,
                    replace(workspace, sessions=(session,)) if claim.reviews is not None else workspace,
                    model,
                    inject_evidence=access == "full",
                    enable_python=access == "python",
                    activity=activity,
                )
                return reviewed.model_copy(update=MappingProxyType({"content_version": version}))
            except (AnalysisResponseError, EvidenceReadError) as error:
                return Examined(
                    execution=execution,
                    observations=(),
                    parts=(),
                    partial=(await workspace.summary(execution.id)).partial,
                    cannot_assess=True,
                    error=str(error),
                    reasoning=str(error),
                    tool_calls=activity.activity.tool_calls,
                )

    completed_reviews, review_error = await collect_reviews(
        examine_executions(claim, sample, read, limited, planned_progress, extractor=extract)
    )
    indexed: Final = MappingProxyType({review.execution.id: review for review in completed_reviews})
    examined: Final = tuple(indexed[execution.id] for execution in sample.executions if execution.id in indexed)
    coverage: Final = base.model_copy(
        update=MappingProxyType(
            {
                "screened": len(examined),
                "partial": sum(
                    review.partial or review.execution.id in workspace.partial_sessions for review in examined
                ),
                "unassessable": sum(review.cannot_assess for review in examined),
                "failed_tasks": sum(bool(review.error) for review in examined),
                "reused": sum(review.reused for review in examined),
                "reusable": reusable,
            }
        )
    )
    observations: Final = tuple(chain.from_iterable(review.observations for review in examined))
    pending: Final = tuple(chain.from_iterable(review.observations for review in examined if not review.consolidated))
    versions: Final = tuple(
        ReviewVersion(execution_id=review.execution.id, content_version=review.content_version)
        for review in examined
        if review.content_version and not review.error
    )

    def assessment(review: Examined) -> RunAssessment:
        supported: Final = tuple(
            observation
            for observation in observations
            if any(
                quote.execution_id == review.execution.id and quote.role == "support" for quote in observation.evidence
            )
        )
        return RunAssessment(
            execution_id=review.execution.id,
            issue_checks=tuple(sorted(frozenset(o.check_id for o in supported if o.kind == "issue"))),
            pattern_checks=tuple(sorted(frozenset(o.check_id for o in supported if o.kind == "pattern"))),
            cannot_assess=review.cannot_assess,
        )

    assessments: Final = tuple(assessment(review) for review in examined)
    if review_error or not pending:
        return Result(
            coverage=coverage,
            assessments=assessments,
            review_versions=() if review_error else versions,
            error="\n\n".join(
                dict.fromkeys(
                    (
                        *((review_error,) if review_error else ()),
                        *(review.error for review in examined if review.error),
                        *sorted(workspace.read_errors),
                    )
                )
            ),
        )
    records: Final = tuple(
        ReviewRecord(
            execution_id=review.execution.id,
            phase="initial",
            content=Extraction(
                observations=review.observations, cannot_assess=review.cannot_assess, reasoning=review.reasoning
            ).model_dump_json(),
        )
        for review in examined
    )
    review_workspace: Final = workspace.with_reviews(records)
    batches: Final = observation_batches(pending)
    grouping: Final = coverage.model_copy(update=MappingProxyType({"grouping_batches": len(batches)}))
    await progress("Grouping observations", grouping)
    try:
        clusters: Final = await parallel_cluster_batches(
            batches, limited, progress, grouping, claim.job.settings.concurrency
        )
    except AnalysisStopped as error:
        return Result(coverage=grouping, assessments=assessments, error=str(error))
    investigating: Final = grouping.model_copy(
        update=MappingProxyType({"grouped_batches": len(batches), "candidates": len(clusters.candidates)})
    )

    async def investigate(item: tuple[int, Candidate]) -> tuple[int, CandidateInvestigation]:
        index, candidate = item
        async with track_activity(
            progress,
            identity=f"investigate:{index}",
            phase="investigate",
            label=candidate.title,
            execution_ids=candidate.execution_ids,
        ) as activity:
            return index, await investigate_context_candidate(
                claim, candidate, review_workspace, limited, access=access, activity=activity
            )

    await progress("Checking original evidence", investigating)
    completed: Final = iter(range(1, len(clusters.candidates) + 1))
    investigated: tuple[tuple[int, CandidateInvestigation], ...] = ()  # rebind-ok: collect candidate results by index
    investigation_error = ""  # rebind-ok: retain verified findings when another candidate cannot finish
    try:
        async with aclosing(
            concurrent_results(tuple(enumerate(clusters.candidates)), investigate, claim.job.settings.concurrency)
        ) as results:
            async for result in results:
                investigated = (*investigated, result)
                await progress(
                    "Checking original evidence",
                    investigating.model_copy(
                        update=MappingProxyType(
                            {
                                "investigated": next(completed),
                                "inconclusive": sum(not item.findings for _, item in investigated),
                                "failed_tasks": coverage.failed_tasks
                                + sum(bool(item.error) for _, item in investigated),
                            }
                        )
                    ),
                )
    except AnalysisStopped as error:
        investigation_error = str(error)
    ordered: Final = tuple(item for _, item in sorted(investigated))
    drafts: Final = tuple(chain.from_iterable(item.findings for item in ordered))
    if not investigation_error:
        await progress("Consolidating findings across runs", investigating)
    consolidated: Final = (
        CandidateInvestigation(error=investigation_error)
        if investigation_error
        else await consolidate_findings(drafts, claim, limited)
    )
    unfinished: Final = frozenset(
        chain.from_iterable(
            candidate.execution_ids
            for candidate, outcome in zip(clusters.candidates, ordered)
            if outcome.error or consolidated.error
        )
    ) | (workspace.partial_sessions if workspace.read_errors else frozenset())
    return Result(
        findings=consolidated.findings,
        assessments=assessments,
        review_versions=()
        if consolidated.error
        else tuple(version for version in versions if version.execution_id not in unfinished),
        error="\n\n".join(
            dict.fromkeys(
                (
                    *(item.error for item in (*examined, *ordered, consolidated) if item.error),
                    *sorted(workspace.read_errors),
                )
            )
        ),
        coverage=investigating.model_copy(
            update=MappingProxyType(
                {
                    "investigated": len(ordered),
                    "inconclusive": sum(not item.findings for item in ordered),
                    "failed_tasks": coverage.failed_tasks + sum(bool(item.error) for item in ordered),
                    "partial": sum(
                        review.partial or review.execution.id in workspace.partial_sessions for review in examined
                    ),
                }
            )
        ),
    )


async def consolidate_findings(
    drafts: tuple[FindingDraft, ...], claim: Claim, model: ModelCall
) -> CandidateInvestigation:
    try:
        return CandidateInvestigation(findings=await reconcile_findings(drafts, claim.findings, model))
    except (AnalysisResponseError, AnalysisStopped) as error:
        return CandidateInvestigation(error=f"Finding consolidation is incomplete: {error}")
