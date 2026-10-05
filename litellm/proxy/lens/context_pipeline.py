import asyncio
from contextlib import aclosing
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
    Claim,
    Coverage,
    Execution,
    FindingDraft,
    ModelRequest,
    ModelResult,
    Record,
    Result,
    RunAssessment,
    Sample,
)

ACCESS: Final[Literal["full", "tools", "python"]] = "python"


class CandidateInvestigation(Record):
    findings: tuple[FindingDraft, ...] = ()
    error: str = ""


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
                return await review_context(
                    claim,
                    session,
                    workspace,
                    model,
                    inject_evidence=access == "full",
                    enable_python=access == "python",
                    activity=activity,
                )
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

    completed_reviews: Final = tuple(
        [review async for review in examine_executions(claim, sample, read, limited, progress, extractor=extract)]
    )
    indexed: Final = MappingProxyType({review.execution.id: review for review in completed_reviews})
    examined: Final = tuple(indexed[execution.id] for execution in sample.executions)
    coverage: Final = base.model_copy(
        update=MappingProxyType(
            {
                "screened": len(examined),
                "partial": sum(
                    review.partial or review.execution.id in workspace.partial_sessions for review in examined
                ),
                "unassessable": sum(review.cannot_assess for review in examined),
            }
        )
    )
    observations: Final = tuple(chain.from_iterable(review.observations for review in examined))

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
    if not observations:
        return Result(
            coverage=coverage,
            assessments=assessments,
            error="\n\n".join(
                dict.fromkeys((*(review.error for review in examined if review.error), *sorted(workspace.read_errors)))
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
    batches: Final = observation_batches(observations)
    grouping: Final = coverage.model_copy(update=MappingProxyType({"grouping_batches": len(batches)}))
    await progress("Grouping observations", grouping)
    clusters: Final = await parallel_cluster_batches(
        batches, limited, progress, grouping, claim.job.settings.concurrency
    )
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
                        }
                    )
                ),
            )
    ordered: Final = tuple(item for _, item in sorted(investigated))
    return Result(
        findings=tuple(chain.from_iterable(item.findings for item in ordered)),
        assessments=assessments,
        error="\n\n".join(
            dict.fromkeys(
                (*(item.error for item in (*examined, *ordered) if item.error), *sorted(workspace.read_errors))
            )
        ),
        coverage=investigating.model_copy(
            update=MappingProxyType(
                {
                    "investigated": len(ordered),
                    "inconclusive": sum(not item.findings for item in ordered),
                    "partial": sum(
                        review.partial or review.execution.id in workspace.partial_sessions for review in examined
                    ),
                }
            )
        ),
    )
