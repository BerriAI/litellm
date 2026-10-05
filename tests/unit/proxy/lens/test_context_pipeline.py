import asyncio
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal

import pytest
from pydantic import BaseModel

from litellm.proxy.lens.agent_review import Findings
from litellm.proxy.lens.agent_runtime import AgentTurn, DialogueTurn
from litellm.proxy.lens.agent_workspace import EvidenceReply, EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import Candidate, Clusters, Extraction, Observation
from litellm.proxy.lens.context_pipeline import (
    investigate_context_candidate,
    parallel_cluster_batches,
    reconcile_candidates,
)
from litellm.proxy.lens.models import (
    Claim,
    Coverage,
    Evidence,
    Execution,
    ExecutionContent,
    FindingDraft,
    ModelRequest,
    ModelResult,
    Sample,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from litellm.proxy.lens.worker import analyze_sample
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, issue_brief, lens


class AgentPrompt(BaseModel):
    stage: str
    supplied: str
    dialogue: tuple[DialogueTurn, ...] = ()
    initial_evidence: tuple[TracePart, ...] = ()


class GroupPrompt(BaseModel):
    candidates: tuple[Candidate, ...]


class AssignedSession(BaseModel):
    execution: Execution


async def ignore_progress(_stage: str, _coverage: Coverage) -> None:
    return None


@pytest.mark.asyncio
async def test_reconciliation_compares_large_candidate_set_once_without_losing_omitted_references() -> None:
    candidates: Final = tuple(
        Candidate(
            check_id="retries",
            title=f"Candidate {index}",
            hypothesis=f"Cause {index}: " + "Complete supporting detail. " * 40,
            execution_ids=(f"run-{index}",),
        )
        for index in range(128)
    )
    calls: Final = SimpleQueue[str]()

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(request.prompt)
        payload: Final = GroupPrompt.model_validate_json(request.prompt)
        assert payload.candidates == tuple(
            candidate.model_copy(update=MappingProxyType({"execution_ids": (f"p{index}",)}))
            for index, candidate in enumerate(candidates)
        )
        return ModelResult(
            content=Clusters(
                candidates=(candidates[0].model_copy(update=MappingProxyType({"execution_ids": ("p0", "p1")})),)
            ).model_dump_json(),
            cost=0,
        )

    result: Final = await reconcile_candidates(candidates, model)
    assert calls.qsize() == 1
    assert result.candidates == (
        candidates[0].model_copy(update=MappingProxyType({"execution_ids": ("run-0", "run-1")})),
        *candidates[2:],
    )


@pytest.mark.asyncio
async def test_production_entrypoint_makes_complete_child_content_available_without_eager_injection() -> None:
    run: Final = execution("real-session", 2)
    root: Final = TracePart(
        execution_id=run.id, span_id="root", name="coordinator", kind="agent", content="Task delivered"
    )
    child: Final = TracePart(
        execution_id=run.id,
        span_id="child",
        parent_span_id="root",
        name="researcher",
        kind="agent",
        content="x" * 9000 + " evidence in the middle " + "x" * 9000,
    )

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=run, parts=(root, child))

    async def model(request: ModelRequest) -> ModelResult:
        assert request.purpose == "extract"
        payload: Final = AgentPrompt.model_validate_json(request.prompt)
        assert payload.initial_evidence == ()
        if not payload.dialogue:
            assert child.content not in request.prompt
            assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="read", execution_id=assigned.id, span_ids=(child.span_id,)),)
                ).model_dump_json(),
                cost=0,
            )
        reply: Final = EvidenceReply.model_validate_json(payload.dialogue[-1].tool_results[0])
        assert reply.parts == (child.model_copy(update=MappingProxyType({"execution_id": "r0"})),)
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=(run,), eligible=1), read, model, ignore_progress)
    assert result.coverage.screened == 1
    assert result.assessments[0].execution_id == run.id
    assert result.findings == ()


@pytest.mark.asyncio
async def test_grouping_overlaps_and_preserves_omitted_observations_in_input_order() -> None:
    observations: Final = tuple(
        Observation(
            check_id="retries",
            summary=summary,
            evidence=(Evidence(execution_id=identity, span_id="span", quote="failure"),),
        )
        for identity, summary in (("first", "Wrong argument"), ("second", "Missing capability"))
    )
    entered: Final = SimpleQueue[str]()
    both_entered: Final = asyncio.Event()
    second_finished: Final = asyncio.Event()
    progress_counts: Final = SimpleQueue[int]()

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = GroupPrompt.model_validate_json(request.prompt)
        if len(payload.candidates) == 1:
            title: Final = payload.candidates[0].title
            entered.put(title)
            if entered.qsize() == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            if title == "Wrong argument":
                await asyncio.wait_for(second_finished.wait(), timeout=1)
            else:
                second_finished.set()
        else:
            assert tuple(candidate.title for candidate in payload.candidates) == (
                "Wrong argument",
                "Missing capability",
            )
        return ModelResult(content=Clusters().model_dump_json(), cost=0)

    async def progress(stage: str, coverage: Coverage) -> None:
        assert stage == "Grouping observations"
        assert coverage.screened == 2
        progress_counts.put(coverage.grouped_batches)

    result: Final = await parallel_cluster_batches(
        tuple((observation,) for observation in observations),
        model,
        progress,
        Coverage(screened=2, grouping_batches=2),
        concurrency=2,
    )
    assert result == Clusters(
        candidates=(
            Candidate(
                check_id="retries", title="Wrong argument", hypothesis="issue: Wrong argument", execution_ids=("first",)
            ),
            Candidate(
                check_id="retries",
                title="Missing capability",
                hypothesis="issue: Missing capability",
                execution_ids=("second",),
            ),
        )
    )
    assert tuple(progress_counts.get_nowait() for _ in range(progress_counts.qsize())) == (1, 2)


@pytest.mark.asyncio
async def test_candidate_investigators_overlap_browse_reviews_and_keep_original_ids_in_order() -> None:
    runs: Final = tuple(
        execution(identity).model_copy(update=MappingProxyType({"root_seen": True}))
        for identity in ("first-session", "second-session")
    )
    entered: Final = SimpleQueue[str]()
    both_entered: Final = asyncio.Event()
    second_finished: Final = asyncio.Event()

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(
            execution=next(run for run in runs if run.id == identity),
            parts=(TracePart(execution_id=identity, span_id="child", name="tool", kind="tool", content="timeout"),),
        )

    async def model(request: ModelRequest) -> ModelResult:
        if request.purpose == "cluster":
            groups: Final = GroupPrompt.model_validate_json(request.prompt)
            return ModelResult(content=Clusters(candidates=groups.candidates).model_dump_json(), cost=0)
        payload: Final = AgentPrompt.model_validate_json(request.prompt)
        if request.purpose == "extract":
            assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
            return ModelResult(
                content=AgentTurn[Extraction](
                    result=Extraction(
                        observations=(
                            Observation(
                                check_id="retries",
                                summary=f"Timeout in {assigned.id}",
                                evidence=(Evidence(execution_id=assigned.id, span_id="child", quote="timeout"),),
                            ),
                        )
                    )
                ).model_dump_json(),
                cost=0,
            )
        candidate: Final = Candidate.model_validate_json(payload.supplied)
        identity: Final = candidate.execution_ids[0]
        if not payload.dialogue:
            entered.put(identity)
            if entered.qsize() == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            if identity == "r0":
                await asyncio.wait_for(second_finished.wait(), timeout=1)
            else:
                second_finished.set()
            return ModelResult(
                content=AgentTurn[Findings](
                    tools=(
                        EvidenceRequest(action="read_reviews", execution_id=identity),
                        EvidenceRequest(action="read", execution_id=identity, span_ids=("child",)),
                    )
                ).model_dump_json(),
                cost=0,
            )
        review_reply: Final = EvidenceReply.model_validate_json(payload.dialogue[-1].tool_results[0])
        assert len(review_reply.reviews) == 1
        assert review_reply.reviews[0].execution_id == identity
        reviewed: Final = Extraction.model_validate_json(review_reply.reviews[0].content)
        assert reviewed.observations[0].evidence == (Evidence(execution_id=identity, span_id="child", quote="timeout"),)
        evidence_reply: Final = EvidenceReply.model_validate_json(payload.dialogue[-1].tool_results[1])
        assert evidence_reply.parts[0].content == "timeout"
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title=candidate.title,
                            description="The attempted operation timed out",
                            check_id="retries",
                            brief=issue_brief("The operation timed out"),
                            evidence=reviewed.observations[0].evidence,
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    initial: Final = lens()
    configured: Final = initial.model_copy(
        update=MappingProxyType({"settings": initial.settings.model_copy(update=MappingProxyType({"concurrency": 2}))})
    )
    claim: Final = Claim(lens_id="lens", job=queue_job(configured, NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=runs, eligible=2), read, model, ignore_progress)
    assert tuple(finding.evidence[0].execution_id for finding in result.findings) == tuple(run.id for run in runs)
    assert tuple(assessment.execution_id for assessment in result.assessments) == tuple(run.id for run in runs)
    assert result.coverage == Coverage(
        eligible=2, selected=2, screened=2, investigated=2, grouping_batches=1, grouped_batches=1, candidates=2
    )


@pytest.mark.asyncio
async def test_candidate_investigator_rejects_fabricated_original_quotes() -> None:
    run: Final = execution("run")
    workspace: Final = EvidenceWorkspace(
        sessions=(
            SessionContent(
                execution=run,
                parts=(TracePart(execution_id=run.id, span_id="child", name="tool", kind="tool", content="timeout"),),
                partial=False,
            ),
        )
    )
    attempts: Final = SimpleQueue[str]()

    async def model(request: ModelRequest) -> ModelResult:
        attempts.put(request.prompt)
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title="Missing evidence",
                            description="This claim is not supported",
                            check_id="retries",
                            brief=issue_brief("The operation timed out"),
                            evidence=(Evidence(execution_id=run.id, span_id="child", quote="invented"),),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate_context_candidate(
        claim,
        Candidate(check_id="retries", title="Timeout", hypothesis="Repeated timeouts", execution_ids=(run.id,)),
        workspace,
        model,
    )
    assert result.findings == ()
    assert "Every evidence quote must exactly match" in result.error
    assert attempts.qsize() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("access", ("full", "tools", "python"))
async def test_investigator_only_injects_candidate_sessions_for_full_access(
    access: Literal["full", "tools", "python"],
) -> None:
    sessions: Final = tuple(
        SessionContent(
            execution=execution(identity),
            parts=(TracePart(execution_id=identity, span_id="span", name="tool", kind="tool", content=content),),
            partial=False,
        )
        for identity, content in (("assigned", "original assigned content"), ("other", "unrelated original content"))
    )
    workspace: Final = EvidenceWorkspace(sessions=sessions)
    candidate: Final = Candidate(
        check_id="retries", title="Candidate", hypothesis="Repeated operation", execution_ids=("assigned",)
    )

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = AgentPrompt.model_validate_json(request.prompt)
        assert payload.initial_evidence == (sessions[0].parts if access == "full" else ())
        assert payload.supplied == candidate.model_dump_json()
        assert sessions[1].parts[0].content not in request.prompt
        return ModelResult(content=AgentTurn[Findings](result=Findings()).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate_context_candidate(claim, candidate, workspace, model, access=access)
    assert result.findings == ()
    assert result.error == ""
