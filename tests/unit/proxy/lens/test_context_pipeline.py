import asyncio
from itertools import chain
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from litellm.proxy.lens.agent_review import Findings
from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceReply, EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import AnalysisResponseError, Candidate, Clusters, Extraction, Observation
from litellm.proxy.lens.context_pipeline import (
    investigate_context_candidate,
    parallel_cluster_batches,
    reconcile_candidates,
)
from litellm.proxy.lens.models import (
    Activity,
    Claim,
    Coverage,
    Evidence,
    Execution,
    ExecutionContent,
    FindingDraft,
    InFlight,
    ModelRequest,
    ModelResult,
    Progress,
    Review,
    Sample,
    ToolCount,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from litellm.proxy.lens.worker import analyze_sample
from tests.unit.proxy.lens.test_agent_runtime import InitialPrompt, ToolReply
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, issue_brief, lens


class GroupPrompt(BaseModel):
    candidates: tuple[Candidate, ...]


class AssignedSession(BaseModel):
    execution: Execution


async def ignore_progress(
    _stage: str | None,
    _coverage: Coverage | None,
    _review: Review | None = None,
    _reading: tuple[InFlight, ...] | None = None,
    _activity: Activity | None = None,
    /,
) -> None:
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
async def test_reconciliation_splits_only_after_overflow_and_preserves_cross_page_merges() -> None:
    candidates: Final = tuple(
        Candidate(check_id="retries", title=cause, hypothesis=cause, execution_ids=(f"run-{index}",))
        for index, cause in enumerate(("cause-a", "cause-b", "cause-c", "cause-d", "cause-b", "cause-d"))
    )
    calls: Final = SimpleQueue[int]()
    activities: Final = SimpleQueue[Activity]()

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = GroupPrompt.model_validate_json(request.prompt)
        calls.put(len(payload.candidates))
        if len(payload.candidates) > 3:
            return ModelResult(content="", cost=0, context_exceeded=True)
        causes: Final = tuple(dict.fromkeys(candidate.hypothesis for candidate in payload.candidates))
        groups: Final = tuple(
            tuple(candidate for candidate in payload.candidates if candidate.hypothesis == cause) for cause in causes
        )
        merged: Final = tuple(
            group[0].model_copy(
                update=MappingProxyType(
                    {"execution_ids": tuple(chain.from_iterable(candidate.execution_ids for candidate in group))}
                )
            )
            for group in groups
            if len(group) > 1
        )
        return ModelResult(content=Clusters(candidates=merged).model_dump_json(), cost=0)

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        if activity is not None:
            activities.put(activity)

    result: Final = await reconcile_candidates(candidates, model, progress)
    assert {candidate.hypothesis: candidate.execution_ids for candidate in result.candidates} == {
        "cause-a": ("run-0",),
        "cause-b": ("run-1", "run-4"),
        "cause-c": ("run-2",),
        "cause-d": ("run-3", "run-5"),
    }
    assert len(result.candidates) == 4
    assert calls.get_nowait() == len(candidates)
    assert any(calls.get_nowait() > 3 for _ in range(calls.qsize()))
    events: Final = tuple(activities.get_nowait() for _ in range(activities.qsize()))
    assert frozenset(event.id for event in events) == frozenset(("reconcile",))
    assert sum(event.finished for event in events) == 1
    assert events[-1].finished
    assert events[-1].operations == ()


@pytest.mark.asyncio
async def test_reconciliation_stops_when_two_candidates_cannot_fit() -> None:
    calls: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(request)
        assert calls.qsize() <= 2
        return ModelResult(content="", cost=0, context_exceeded=True)

    candidates: Final = tuple(
        Candidate(check_id="retries", title=f"Cause {index}", hypothesis="Large summary", execution_ids=(str(index),))
        for index in range(2)
    )
    with pytest.raises(AnalysisResponseError, match="smallest candidate comparison exceeds"):
        await reconcile_candidates(candidates, model)
    assert 1 <= calls.qsize() <= 2


@pytest.mark.asyncio
async def test_production_entrypoint_makes_complete_child_content_available_without_eager_injection() -> None:
    reports: Final = SimpleQueue[Progress]()
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
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        assert payload.initial_evidence == ()
        if len(request.messages) == 2:
            assert all(child.content not in message.content for message in request.messages)
            assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="read", execution_id=assigned.id, span_ids=(child.span_id,)),)
                ).model_dump_json(),
                cost=0,
            )
        reply: Final = EvidenceReply.model_validate_json(
            ToolReply.model_validate_json(request.messages[-1].content).tool_results[0]
        )
        assert reply.parts == (child.model_copy(update=MappingProxyType({"execution_id": "r0"})),)
        return ModelResult(
            content=AgentTurn[Extraction](result=Extraction(reasoning="Recorded task completed.")).model_dump_json(),
            cost=0,
        )

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        review: Review | None = None,
        reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        reports.put(Progress(stage=stage, coverage=coverage, review=review, reading=reading, activity=activity))

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=(run,), eligible=1), read, model, progress)
    assert result.coverage.screened == 1
    assert result.assessments[0].execution_id == run.id
    assert result.findings == ()
    events: Final = tuple(reports.get_nowait() for _ in range(reports.qsize()))
    reviews: Final = tuple(event.review for event in events if event.review is not None)
    assert len(reviews) == 1
    assert reviews[0].execution_id == run.id
    assert reviews[0].reasoning == "Recorded task completed."
    assert reviews[0].tool_calls == (ToolCount(name="read", calls=1),)
    assert reviews[0].spans == ()
    activities: Final = tuple(event.activity for event in events if event.activity is not None)
    assert frozenset(activity.phase for activity in activities) == frozenset(("load", "review"))
    assert all(activity.execution_ids == (run.id,) for activity in activities)
    assert all(child.content not in activity.model_dump_json() for activity in activities)
    assert tuple(activity.phase for activity in activities if activity.finished) == ("load", "review")
    assert any(activity.phase == "review" and activity.operations == ("read",) for activity in activities)
    assert any(event.reading and event.reading[0].execution_id == run.id for event in events)


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
    activities: Final = SimpleQueue[Activity]()

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

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if _activity is not None:
            activities.put(_activity)
        if coverage is None:
            return
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
    events: Final = tuple(activities.get_nowait() for _ in range(activities.qsize()))
    assert frozenset(event.phase for event in events) == frozenset(("group", "reconcile"))
    assert frozenset(event.id for event in events if event.finished) == frozenset(("group:0", "group:1", "reconcile"))


@pytest.mark.asyncio
async def test_initial_group_overflow_preserves_every_observation_and_execution_reference() -> None:
    observations: Final = tuple(
        Observation(
            check_id="retries",
            summary=f"Distinct cause {index}",
            evidence=(Evidence(execution_id=f"run-{index}", span_id="span", quote="failure"),),
        )
        for index in range(5)
    )
    calls: Final = SimpleQueue[int]()
    progress_counts: Final = SimpleQueue[int]()

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = GroupPrompt.model_validate_json(request.prompt)
        calls.put(len(payload.candidates))
        if len(payload.candidates) > 2:
            return ModelResult(content="", cost=0, context_exceeded=True)
        return ModelResult(content=Clusters().model_dump_json(), cost=0)

    async def progress(
        _stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if coverage is not None:
            progress_counts.put(coverage.grouped_batches)

    result: Final = await parallel_cluster_batches(
        (observations,), model, progress, Coverage(screened=5, grouping_batches=1), concurrency=2
    )
    assert result == Clusters(
        candidates=tuple(
            Candidate(
                check_id="retries",
                title=observation.summary,
                hypothesis=f"issue: {observation.summary}",
                execution_ids=(observation.evidence[0].execution_id,),
            )
            for observation in observations
        )
    )
    assert calls.get_nowait() == len(observations)
    assert tuple(progress_counts.get_nowait() for _ in range(progress_counts.qsize())) == (1,)


@pytest.mark.asyncio
async def test_candidate_investigators_overlap_browse_reviews_and_keep_original_ids_in_order() -> None:
    activities: Final = SimpleQueue[Activity]()
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
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
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
        if len(request.messages) == 2:
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
        review_reply: Final = EvidenceReply.model_validate_json(
            ToolReply.model_validate_json(request.messages[-1].content).tool_results[0]
        )
        assert len(review_reply.reviews) == 1
        assert review_reply.reviews[0].execution_id == identity
        reviewed: Final = Extraction.model_validate_json(review_reply.reviews[0].content)
        assert reviewed.observations[0].evidence == (Evidence(execution_id=identity, span_id="child", quote="timeout"),)
        evidence_reply: Final = EvidenceReply.model_validate_json(
            ToolReply.model_validate_json(request.messages[-1].content).tool_results[1]
        )
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

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        if activity is not None:
            activities.put(activity)

    claim: Final = Claim(lens_id="lens", job=queue_job(configured, NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=runs, eligible=2), read, model, progress)
    assert tuple(finding.evidence[0].execution_id for finding in result.findings) == tuple(run.id for run in runs)
    assert tuple(assessment.execution_id for assessment in result.assessments) == tuple(run.id for run in runs)
    assert result.coverage == Coverage(
        eligible=2, selected=2, screened=2, investigated=2, grouping_batches=1, grouped_batches=1, candidates=2
    )
    events: Final = tuple(activities.get_nowait() for _ in range(activities.qsize()))
    final_checks: Final = tuple(event for event in events if event.phase == "investigate" and event.finished)
    assert frozenset(event.execution_ids for event in final_checks) == frozenset((run.id,) for run in runs)
    assert all(
        frozenset(event.tool_calls)
        == frozenset((ToolCount(name="read_reviews", calls=1), ToolCount(name="read", calls=1)))
        for event in final_checks
    )
    assert all(event.operations == () for event in final_checks)


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
@pytest.mark.parametrize("failure", ("source", "model", "content"))
async def test_candidate_distinguishes_gateway_schema_failure_from_malformed_model_output(failure: str) -> None:
    run: Final = execution("run")
    calls: Final = SimpleQueue[ModelRequest]()
    reads: Final = SimpleQueue[str]()

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        reads.put(identity)
        if failure == "content":
            return ExecutionContent(execution=run, parts=(), next_cursor="repeat")
        return ExecutionContent.model_validate({"execution": run.model_dump(), "parts": "malformed gateway evidence"})

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(request)
        if failure == "model":
            return ModelResult(content="raw-private-model-output", cost=0)
        if failure == "content" and calls.qsize() == 2:
            assert "Could not verify this citation" in request.messages[-1].content
            return ModelResult(content=AgentTurn[Findings](result=Findings()).model_dump_json(), cost=0)
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title="The tool timed out",
                            description="The operation did not complete",
                            check_id="retries",
                            brief=issue_brief("The operation timed out"),
                            evidence=(Evidence(execution_id=run.id, span_id="child", quote="timeout"),),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    workspace: Final = EvidenceWorkspace(sessions=(SessionContent(execution=run, parts=(), partial=False),), read=read)
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    candidate: Final = Candidate(
        check_id="retries", title="Timeout", hypothesis="Repeated timeouts", execution_ids=(run.id,)
    )
    if failure == "source":
        with pytest.raises(ValidationError) as raised:
            await investigate_context_candidate(claim, candidate, workspace, model)
        assert raised.value.errors()[0]["loc"] == ("parts",)
        assert calls.qsize() == 1
        assert reads.get_nowait() == run.id
    elif failure == "content":
        incomplete: Final = await investigate_context_candidate(claim, candidate, workspace, model)
        assert incomplete.findings == ()
        assert incomplete.error == ""
        assert any("repeated a pagination cursor" in error for error in workspace.read_errors)
        assert calls.qsize() == 2
        assert tuple(reads.get_nowait() for _ in range(reads.qsize())) == (run.id, run.id)
    else:
        result: Final = await investigate_context_candidate(claim, candidate, workspace, model)
        assert result.findings == ()
        assert "response invalid after 2 attempts" in result.error
        assert "raw-private-model-output" not in result.error
        assert calls.qsize() == 2
        assert reads.empty()


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
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        assert payload.initial_evidence == (sessions[0].parts if access == "full" else ())
        assert payload.supplied == candidate.model_dump_json()
        assert all(sessions[1].parts[0].content not in message.content for message in request.messages)
        return ModelResult(content=AgentTurn[Findings](result=Findings()).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate_context_candidate(claim, candidate, workspace, model, access=access)
    assert result.findings == ()
    assert result.error == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "supported_finding"),
    (
        ("invalid", True),
        ("context", True),
        ("invalid", False),
        ("cursor", True),
        ("span", True),
        ("eof", True),
        ("cursor", False),
    ),
)
async def test_failed_session_review_preserves_other_results_and_reports_its_error(
    failure: str, supported_finding: bool
) -> None:
    runs: Final = tuple(
        execution(identity).model_copy(update=MappingProxyType({"root_seen": True})) for identity in ("failed", "valid")
    )
    reviews: Final = SimpleQueue[Review]()

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        if identity == "failed" and failure == "cursor":
            return ExecutionContent(execution=runs[0], parts=(), next_cursor="repeat")
        if identity == "failed" and failure in ("span", "eof"):
            return ExecutionContent(
                execution=runs[0],
                parts=(
                    TracePart(
                        execution_id=identity,
                        span_id="child",
                        name="tool",
                        kind="tool",
                        content="x" * 8000 if _offset == 1 else "",
                        truncated=True,
                    ),
                )
                if _offset == 1 or failure == "eof"
                else (),
            )
        return ExecutionContent(
            execution=next(run for run in runs if run.id == identity),
            parts=(TracePart(execution_id=identity, span_id="child", name="tool", kind="tool", content="timeout"),),
        )

    async def model(request: ModelRequest) -> ModelResult:
        if request.purpose == "cluster":
            groups: Final = GroupPrompt.model_validate_json(request.prompt)
            return ModelResult(content=Clusters(candidates=groups.candidates).model_dump_json(), cost=0)
        if "Compact this analysis conversation" in request.messages[-1].content:
            return ModelResult(content="", cost=0, context_exceeded=True)
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        if request.purpose == "extract":
            assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
            if assigned.name == "failed":
                if failure in ("cursor", "span", "eof"):
                    if len(request.messages) > 2:
                        reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
                        problem: Final = EvidenceReply.model_validate_json(reply.tool_results[0])
                        assert "Original trace" in problem.error
                        return ModelResult(
                            content=AgentTurn[Extraction](
                                result=Extraction(cannot_assess=True, reasoning=problem.error)
                            ).model_dump_json(),
                            cost=0,
                        )
                    return ModelResult(
                        content=AgentTurn[Extraction](
                            tools=(
                                EvidenceRequest(
                                    action="read", execution_id=assigned.id, char_start=1 if failure == "span" else 0
                                ),
                            )
                        ).model_dump_json(),
                        cost=0,
                    )
                return ModelResult(
                    content="raw-private-response-sentinel", cost=0, context_exceeded=failure == "context"
                )
            return ModelResult(
                content=AgentTurn[Extraction](
                    result=Extraction(
                        observations=(
                            Observation(
                                check_id="retries",
                                summary="The tool timed out",
                                evidence=(Evidence(execution_id=assigned.id, span_id="child", quote="timeout"),),
                            ),
                        )
                        if supported_finding
                        else ()
                    )
                ).model_dump_json(),
                cost=0,
            )
        candidate: Final = Candidate.model_validate_json(payload.supplied)
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title="The tool timed out",
                            description="A recorded operation timed out",
                            check_id="retries",
                            brief=issue_brief("The operation timed out"),
                            evidence=(
                                Evidence(execution_id=candidate.execution_ids[0], span_id="child", quote="timeout"),
                            ),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if review is not None:
            reviews.put(review)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=runs, eligible=2), read, model, progress)
    assert tuple(finding.evidence[0].execution_id for finding in result.findings) == (
        ("valid",) if supported_finding else ()
    )
    assert {assessment.execution_id: assessment.cannot_assess for assessment in result.assessments} == {
        "failed": True,
        "valid": False,
    }
    assert result.coverage.screened == 2
    assert result.coverage.unassessable == 1
    assert result.coverage.partial == int(failure in ("cursor", "span", "eof"))
    assert result.coverage.investigated == int(supported_finding)
    assert result.error
    assert "raw-private-response-sentinel" not in result.error
    assert ("context window" in result.error) is (failure == "context")
    if failure in ("cursor", "span", "eof"):
        assert "Original trace" in result.error
    completed: Final = tuple(reviews.get_nowait() for _ in range(reviews.qsize()))
    assert {review.execution_id: review.cannot_assess for review in completed} == {"failed": True, "valid": False}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "action"),
    (("review", "read"), ("review", "search"), ("review", "catalog"), ("investigate", "read"), ("empty", "read")),
)
@pytest.mark.parametrize("already_partial", (False, True))
async def test_late_content_failure_refreshes_partial_coverage_without_changing_the_source_verdict(
    phase: str, action: Literal["read", "search", "catalog"], already_partial: bool
) -> None:
    runs: Final = tuple(
        execution(identity).model_copy(
            update=MappingProxyType({"root_seen": identity == "source" or not already_partial})
        )
        for identity in ("source", "reader")
    )
    source_reviewed: Final = asyncio.Event()
    evidence: Final = Evidence(execution_id="r0" if phase == "investigate" else "r1", span_id="span", quote="timeout")
    observation: Final = Observation(check_id="retries", summary="The tool timed out", evidence=(evidence,))
    finding: Final = FindingDraft(
        title=observation.summary,
        description="A recorded operation timed out",
        check_id="retries",
        brief=issue_brief("The operation timed out"),
        evidence=(evidence,),
    )
    tool_call: Final = AgentTurn[Extraction](
        tools=(EvidenceRequest(action=action, execution_id="r0", query="timeout"),)
    ).model_dump_json()

    async def read(identity: str, cursor: str, _offset: int) -> ExecutionContent:
        if cursor:
            assert source_reviewed.is_set()
        return ExecutionContent(
            execution=next(run for run in runs if run.id == identity),
            parts=(TracePart(execution_id=identity, span_id="span", name="tool", kind="tool", content="timeout"),),
            next_cursor="repeat" if identity == "source" else None,
        )

    async def model(request: ModelRequest) -> ModelResult:
        if request.purpose == "cluster":
            groups: Final = GroupPrompt.model_validate_json(request.prompt)
            return ModelResult(content=Clusters(candidates=groups.candidates).model_dump_json(), cost=0)
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        if len(request.messages) > 2:
            reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
            failure: Final = EvidenceReply.model_validate_json(reply.tool_results[0])
            assert "repeated a pagination cursor" in failure.error
            assert "r0" in failure.error and "source" in failure.error
            assert "narrower" in failure.error and "other evidence" in failure.error
        if request.purpose == "extract":
            assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
            if assigned.name == "reader":
                await source_reviewed.wait()
                if phase != "investigate" and len(request.messages) == 2:
                    return ModelResult(content=tool_call, cost=0)
            observes: Final = (assigned.name == "source" and phase == "investigate") or (
                assigned.name == "reader" and phase == "review"
            )
            return ModelResult(
                content=AgentTurn[Extraction](
                    result=Extraction(observations=(observation,) if observes else ())
                ).model_dump_json(),
                cost=0,
            )
        if phase == "investigate" and len(request.messages) == 2:
            return ModelResult(content=tool_call, cost=0)
        return ModelResult(
            content=AgentTurn[Findings](result=Findings(findings=(finding,))).model_dump_json(),
            cost=0,
        )

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if review is not None and review.execution_id == "source":
            assert not review.cannot_assess
            source_reviewed.set()

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=runs, eligible=2), read, model, progress)
    assert tuple(item.evidence[0].execution_id for item in result.findings) == (
        () if phase == "empty" else ("source" if phase == "investigate" else "reader",)
    )
    assert result.coverage.partial == 1 + int(already_partial)
    assert result.coverage.screened == 2
    assert result.coverage.investigated == int(phase != "empty")
    assert result.coverage.unassessable == 0
    assert "repeated a pagination cursor" in result.error
    assert "source" in result.error
    assert {assessment.execution_id: assessment.cannot_assess for assessment in result.assessments} == {
        "source": False,
        "reader": False,
    }


@pytest.mark.asyncio
async def test_cross_session_observations_attribute_assessments_and_candidates_only_to_supporting_runs() -> None:
    runs: Final = tuple(
        execution(identity).model_copy(update=MappingProxyType({"root_seen": True}))
        for identity in ("assigned", "affected", "healthy")
    )
    reviews: Final = SimpleQueue[Review]()
    candidates: Final = SimpleQueue[Candidate]()
    comparisons: Final[tuple[tuple[Literal["issue", "pattern"], str, str], ...]] = (
        ("issue", "r1", "r0"),
        ("pattern", "r2", "r1"),
    )

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(
            execution=next(run for run in runs if run.id == identity),
            parts=(
                TracePart(execution_id=identity, span_id="span", name="tool", kind="tool", content="recorded behavior"),
            ),
        )

    async def model(request: ModelRequest) -> ModelResult:
        if request.purpose == "cluster":
            groups: Final = GroupPrompt.model_validate_json(request.prompt)
            return ModelResult(content=Clusters(candidates=groups.candidates).model_dump_json(), cost=0)
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        if request.purpose == "extract":
            assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
            return ModelResult(
                content=AgentTurn[Extraction](
                    result=Extraction(
                        observations=tuple(
                            Observation(
                                check_id="retries",
                                kind=kind,
                                summary=kind,
                                evidence=(
                                    Evidence(execution_id=support, span_id="span", quote="recorded behavior"),
                                    Evidence(
                                        execution_id=counterexample,
                                        span_id="span",
                                        quote="recorded behavior",
                                        role="counterexample",
                                    ),
                                ),
                            )
                            for kind, support, counterexample in comparisons
                        )
                        if assigned.name == "assigned"
                        else ()
                    )
                ).model_dump_json(),
                cost=0,
            )
        candidates.put(Candidate.model_validate_json(payload.supplied))
        return ModelResult(content=AgentTurn[Findings](result=Findings()).model_dump_json(), cost=0)

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if review is not None:
            reviews.put(review)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=runs, eligible=3), read, model, progress)
    assert {
        assessment.execution_id: (assessment.issue_checks, assessment.pattern_checks)
        for assessment in result.assessments
    } == {"assigned": ((), ()), "affected": (("retries",), ()), "healthy": ((), ("retries",))}
    grouped: Final = tuple(candidates.get_nowait() for _ in range(candidates.qsize()))
    assert {candidate.kind: candidate.execution_ids for candidate in grouped} == {"issue": ("r1",), "pattern": ("r2",)}
    completed: Final = tuple(reviews.get_nowait() for _ in range(reviews.qsize()))
    assert next(review for review in completed if review.execution_id == "assigned").verdicts == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("cancelled", "transport", "budget"))
@pytest.mark.parametrize("boundary", ("model", "source"))
async def test_investigation_propagates_systemic_review_failures(failure: str, boundary: str) -> None:
    request: Final = httpx.Request("POST", "https://worker.invalid/model")
    error: Final = (
        asyncio.CancelledError()
        if failure == "cancelled"
        else httpx.ConnectError("worker unavailable")
        if failure == "transport"
        else httpx.HTTPStatusError("budget exhausted", request=request, response=httpx.Response(402, request=request))
    )
    run: Final = execution("run").model_copy(update=MappingProxyType({"root_seen": True}))
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        if boundary == "source":
            raise error
        return ExecutionContent(
            execution=run,
            parts=(TracePart(execution_id=identity, span_id="span", name="tool", kind="tool", content="recorded"),),
        )

    async def model(_request: ModelRequest) -> ModelResult:
        if boundary == "source":
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        raise error

    with pytest.raises(type(error)) as raised:
        await analyze_sample(claim, Sample(executions=(run,), eligible=1), read, model, ignore_progress)
    assert raised.value is error


@pytest.mark.asyncio
async def test_metadata_only_review_does_not_fetch_traces_or_treat_unloaded_content_as_missing() -> None:
    run: Final = execution("run", 17).model_copy(update=MappingProxyType({"root_seen": True}))
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    reviews: Final = SimpleQueue[Review]()
    activities: Final = SimpleQueue[Activity]()

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        pytest.fail("An unrequested trace was fetched to construct the review or its preview")

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        assert payload.initial_evidence == ()
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        if review is not None:
            reviews.put(review)
        if activity is not None:
            activities.put(activity)

    result: Final = await analyze_sample(claim, Sample(executions=(run,), eligible=1), read, model, progress)
    assert result.coverage.screened == 1
    assert result.coverage.partial == result.coverage.unassessable == 0
    assert len(result.assessments) == 1
    assert not result.assessments[0].cannot_assess
    assert reviews.get_nowait().spans == ()
    preparation: Final = tuple(
        activity
        for activity in (activities.get_nowait() for _ in range(activities.qsize()))
        if activity.phase == "load"
    )
    assert preparation[-1].finished
    assert all(activity.operations == activity.tool_calls == () for activity in preparation)
