import asyncio
from contextlib import aclosing
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from pydantic import ValidationError

from .agent_review import FINDINGS_TASK, Findings, review_context, validate_findings
from .agent_runtime import run_agent
from .agent_workspace import EvidenceWorkspace, ReviewRecord, load_workspace
from .analysis import (
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
    candidate_size,
    concurrent_results,
    examine_executions,
    merge_candidates,
    observation_batches,
    partition_items,
    validation_details,
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

ACCESS: Final[Literal["full", "tools", "python"]] = "tools"


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
                execution_ids=tuple(sorted(frozenset(quote.execution_id for quote in observation.evidence))),
            )
            for observation in observations
        )
        merged, preserved = await merge_candidates(incoming, 0, model)
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
    return await reconcile_candidates(candidates, model)


async def reconcile_candidates(candidates: tuple[Candidate, ...], model: ModelCall) -> Clusters:
    registry: tuple[Candidate, ...] = ()  # rebind-ok: reconcile each batch with every earlier matching candidate
    ordered: Final = tuple(sorted(candidates, key=lambda candidate: (candidate.check_id, candidate.kind)))
    for incoming in partition_items(ordered, candidate_size, 8000):
        kinds = frozenset((candidate.check_id, candidate.kind) for candidate in incoming)
        matching = tuple(candidate for candidate in registry if (candidate.check_id, candidate.kind) in kinds)
        unrelated = tuple(candidate for candidate in registry if (candidate.check_id, candidate.kind) not in kinds)
        carried: tuple[Candidate, ...] = incoming
        retained: tuple[Candidate, ...] = ()
        for prior in partition_items(matching, candidate_size, 16000) or ((),):
            continued, settled = await merge_candidates((*prior, *carried), len(prior), model)
            carried = continued
            retained = (*retained, *settled)
        registry = (*unrelated, *retained, *carried)
    return Clusters(candidates=registry)


async def investigate_context_candidate(
    claim: Claim,
    candidate: Candidate,
    workspace: EvidenceWorkspace,
    model: ModelCall,
    *,
    access: Literal["full", "tools", "python"] = ACCESS,
) -> CandidateInvestigation:
    try:
        response: Final = await run_agent(
            stage="context_investigation",
            task=FINDINGS_TASK
            + "\nInvestigate the supplied candidate against original evidence, including counterexamples. "
            "Reviewer records contain the initial observations and exact evidence references. Use read_reviews "
            "for the candidate's sessions and search_reviews to compare other sessions when useful. You can "
            "inspect every sampled session and its nested agents. Preserve distinct supported causes if the "
            "candidate conflates them. Return every supported finding for this candidate, or an empty findings "
            "list if the evidence does not support it.",
            purpose="investigate",
            claim=claim,
            workspace=workspace,
            model=model,
            schema=Findings,
            initial_evidence=(
                tuple(part for part in workspace.parts if part.execution_id in candidate.execution_ids)
                if access == "full"
                else ()
            ),
            supplied=candidate.model_dump_json(),
            validate=lambda findings: validate_findings(claim, workspace, findings),
            enable_python=access == "python",
        )
        return CandidateInvestigation(findings=response.findings)
    except (ValidationError, AnalysisResponseError) as error:
        return CandidateInvestigation(
            error=validation_details(error) if isinstance(error, ValidationError) else str(error)
        )


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
    workspace: Final = await load_workspace(sample, read, claim.job.settings.concurrency)
    slots: Final = asyncio.Semaphore(claim.job.settings.concurrency)

    async def limited(request: ModelRequest) -> ModelResult:
        async with slots:
            return await model(request)

    async def extract(claim: Claim, execution: Execution, _read: ReadContent, model: ModelCall) -> Examined:
        session: Final = next(session for session in workspace.sessions if session.execution.id == execution.id)
        return await review_context(
            claim, session, workspace, model, inject_evidence=access == "full", enable_python=access == "python"
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
                "partial": sum(review.partial for review in examined),
                "unassessable": sum(review.cannot_assess for review in examined),
            }
        )
    )
    assessments: Final = tuple(
        RunAssessment(
            execution_id=review.execution.id,
            issue_checks=tuple(sorted(frozenset(o.check_id for o in review.observations if o.kind == "issue"))),
            pattern_checks=tuple(sorted(frozenset(o.check_id for o in review.observations if o.kind == "pattern"))),
            cannot_assess=review.cannot_assess,
        )
        for review in examined
    )
    observations: Final = tuple(chain.from_iterable(review.observations for review in examined))
    if not observations:
        return Result(coverage=coverage, assessments=assessments)
    records: Final = tuple(
        ReviewRecord(
            execution_id=review.execution.id,
            phase="initial",
            content=Extraction(observations=review.observations, cannot_assess=review.cannot_assess).model_dump_json(),
        )
        for review in examined
    )
    review_workspace: Final = workspace.model_copy(update=MappingProxyType({"reviews": records}))
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
        return index, await investigate_context_candidate(claim, candidate, review_workspace, limited, access=access)

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
        error="\n\n".join(dict.fromkeys(item.error for item in (*examined, *ordered) if item.error)),
        coverage=investigating.model_copy(
            update=MappingProxyType(
                {"investigated": len(ordered), "inconclusive": sum(not item.findings for item in ordered)}
            )
        ),
    )
