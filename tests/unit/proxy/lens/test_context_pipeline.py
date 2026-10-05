import asyncio
from itertools import chain
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal

import pytest
from pydantic import BaseModel

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
    assert tuple((span.span_id, span.preview) for span in reviews[0].spans) == ((root.span_id, root.content),)
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
