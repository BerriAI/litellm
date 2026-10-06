import asyncio
from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.lens.agent_review import review_context
from litellm.proxy.lens.agent_runtime import AgentTurn, JournalReply
from litellm.proxy.lens.agent_workspace import EvidenceReply, EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import Extraction, Observation, review_of
from litellm.proxy.lens.models import Claim, Evidence, ExecutionContent, ModelRequest, ModelResult, TracePart
from litellm.proxy.lens.state import queue_job
from tests.unit.proxy.lens.test_agent_runtime import InitialPrompt, ToolReply
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens


@pytest.mark.parametrize("inject_evidence", (False, True))
@pytest.mark.asyncio
async def test_context_review_reads_and_cites_original_evidence_with_optional_initial_injection(
    inject_evidence: bool,
) -> None:
    quote: Final = "unique original failure"
    part: Final = TracePart(
        execution_id="run",
        span_id="child",
        parent_span_id="parent",
        name="child",
        kind="tool",
        content="original prefix " * 2000 + quote + " original suffix" * 2000,
    )
    unrelated: Final = TracePart(
        execution_id="run", span_id="root", name="root", kind="agent", content="unrequested root content " * 5000
    )
    session: Final = SessionContent(execution=execution("run"), parts=(unrelated, part), partial=False)
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id=claim.job.settings.analysis_checks[0].id,
                summary="Recorded failure",
                evidence=(Evidence(execution_id="run", span_id="child", quote=quote),),
            ),
        )
    )
    turns: Final = iter((0, 1))

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        if next(turns) == 0:
            assert any(part.content in message.content for message in request.messages) is inject_evidence
            assert payload.initial_evidence == (session.parts if inject_evidence else ())
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(
                        EvidenceRequest(
                            action="read",
                            execution_id="run",
                            span_ids=("child",),
                        ),
                    )
                ).model_dump_json(),
                cost=0,
            )
        reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
        assert EvidenceReply.model_validate_json(reply.tool_results[0]).parts == (part,)
        return ModelResult(content=AgentTurn[Extraction](result=expected).model_dump_json(), cost=0)

    result: Final = await review_context(
        claim,
        session,
        EvidenceWorkspace(sessions=(session,)),
        model,
        inject_evidence=inject_evidence,
    )
    assert result.observations == expected.observations
    assert result.parts == (part.model_copy(update=MappingProxyType({"content": quote, "truncated": True})),)


@pytest.mark.asyncio
async def test_cross_session_citations_keep_original_provenance_and_do_not_appear_under_the_assigned_trace() -> None:
    assigned: Final = execution("assigned")
    other: Final = execution("other")
    root: Final = TracePart(
        execution_id=assigned.id, span_id="root", name="root", kind="agent", content="Assigned task"
    )
    related: Final = TracePart(
        execution_id=other.id, span_id="other-span", name="tool", kind="tool", content="Related failure"
    )
    session: Final = SessionContent(execution=assigned, parts=(root,), partial=False)
    workspace: Final = EvidenceWorkspace(
        sessions=(session, SessionContent(execution=other, parts=(related,), partial=False))
    )
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = Extraction(
        reasoning="Compared the assigned task with a related failure.",
        observations=(
            Observation(
                check_id="retries",
                summary="Related failure",
                evidence=(
                    Evidence(execution_id=other.id, span_id=related.span_id, quote=related.content),
                    Evidence(execution_id=assigned.id, span_id=root.span_id, quote=root.content, role="counterexample"),
                ),
            ),
        ),
    )

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content=AgentTurn[Extraction](result=result).model_dump_json(), cost=0)

    examined: Final = await review_context(claim, session, workspace, model)
    review: Final = review_of(examined, claim.job.settings.model, 0, NOW)
    assert frozenset(examined.parts) == frozenset(
        part.model_copy(update=MappingProxyType({"truncated": True})) for part in (root, related)
    )
    assert examined.observations == result.observations
    assert review.execution_id == assigned.id and review.trace_id == assigned.trace_id
    assert tuple(span.span_id for span in review.spans) == (root.span_id,)
    assert review.reasoning == result.reasoning
    assert review.verdicts == ()


@pytest.mark.asyncio
async def test_observation_with_only_counterexamples_requires_supporting_evidence() -> None:
    part: Final = TracePart(execution_id="run", span_id="span", name="tool", kind="tool", content="recorded behavior")
    session: Final = SessionContent(execution=execution("run"), parts=(part,), partial=False)
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    roles: Final = iter(("counterexample", "support"))

    async def model(request: ModelRequest) -> ModelResult:
        role: Final = next(roles)
        if role == "support":
            assert "requires supporting original evidence" in request.messages[-1].content
        return ModelResult(
            content=AgentTurn[Extraction](
                result=Extraction(
                    observations=(
                        Observation(
                            check_id="retries",
                            summary="Recorded behavior",
                            evidence=(Evidence(execution_id="run", span_id="span", quote=part.content, role=role),),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    result: Final = await review_context(claim, session, EvidenceWorkspace(sessions=(session,)), model)
    assert result.observations[0].evidence == (Evidence(execution_id="run", span_id="span", quote=part.content),)


@pytest.mark.asyncio
async def test_unreadable_citation_can_be_repaired_without_discarding_the_healthy_review() -> None:
    runs: Final = (execution("healthy"), execution("damaged"))
    sessions: Final = tuple(SessionContent(execution=run, partial=False) for run in runs)
    part: Final = TracePart(
        execution_id="healthy", span_id="span", name="tool", kind="tool", content="Recorded failure"
    )
    turns: Final = iter(("damaged", "healthy"))

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return (
            ExecutionContent(execution=runs[0], parts=(part,))
            if identity == "healthy"
            else ExecutionContent(execution=runs[1], parts=(), next_cursor="repeat")
        )

    async def model(request: ModelRequest) -> ModelResult:
        identity: Final = next(turns)
        if identity == "healthy":
            assert "Could not verify this citation" in request.messages[-1].content
            assert "damaged" in request.messages[-1].content
        return ModelResult(
            content=AgentTurn[Extraction](
                result=Extraction(
                    observations=(
                        Observation(
                            check_id="retries",
                            summary=part.content,
                            evidence=(Evidence(execution_id=identity, span_id="span", quote=part.content),),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    workspace: Final = EvidenceWorkspace(sessions=sessions, read=read)
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await review_context(claim, sessions[0], workspace, model)
    assert not result.cannot_assess and not result.partial
    assert result.observations[0].evidence == (Evidence(execution_id="healthy", span_id="span", quote=part.content),)
    assert workspace.partial_sessions == {"damaged"}


@pytest.mark.asyncio
async def test_review_previews_use_verified_quotes_without_rereading_mutable_sources() -> None:
    run: Final = execution("run")
    session: Final = SessionContent(execution=run, partial=False)
    part: Final = TracePart(
        execution_id=run.id, span_id="span", parent_span_id="root", name="tool", kind="tool", content="first then last"
    )
    reads: Final = iter((part, part))
    quotes: Final = ("first", "last")
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id="retries",
                summary="Two verified excerpts",
                evidence=tuple(Evidence(execution_id=run.id, span_id=part.span_id, quote=quote) for quote in quotes),
            ),
        )
    )

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=run, parts=(next(reads),))

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content=AgentTurn[Extraction](result=expected).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    workspace: Final = EvidenceWorkspace(sessions=(session,), read=read)
    result: Final = await review_context(claim, session, workspace, model)
    assert result.observations == expected.observations
    assert result.parts == (
        part.model_copy(
            update=MappingProxyType({"content": "first\n[... content omitted ...]\nlast", "truncated": True})
        ),
    )


@pytest.mark.asyncio
async def test_format_repair_keeps_citation_feedback_and_tools_available_until_evidence_is_valid() -> None:
    parts: Final = (
        TracePart(execution_id="run", span_id="tool", name="tool", kind="tool", content="Original timeout"),
        TracePart(execution_id="run", span_id="final", name="final", kind="agent", content="Recovered later"),
    )
    session: Final = SessionContent(execution=execution("run"), parts=parts, partial=False)
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id="retries",
                summary="Timeout followed by recovery",
                evidence=tuple(
                    Evidence(execution_id=part.execution_id, span_id=part.span_id, quote=part.content) for part in parts
                ),
            ),
        )
    )
    invalid: Final = expected.model_copy(
        update={
            "observations": (
                expected.observations[0].model_copy(
                    update={
                        "evidence": (
                            Evidence(execution_id="run", span_id="tool", quote="private invented text"),
                            Evidence(execution_id="run", span_id="tool", quote=parts[1].content),
                        )
                    }
                ),
            )
        }
    )
    turns: Final = iter(range(6))

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 0:
            return ModelResult(content=invalid.model_dump_json(), cost=0)
        if turn == 1:
            assert request.messages[-1].role == "system"
            assert "response_schema" in request.messages[-1].content
            return ModelResult(content=AgentTurn[Extraction](result=invalid).model_dump_json(), cost=0)
        if turn == 2:
            feedback: Final = request.messages[-1]
            assert feedback.role == "system"
            assert "result.observations[0].evidence[0]" in feedback.content
            assert "result.observations[0].evidence[1]" in feedback.content
            assert "private invented text" not in feedback.content
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        if turn == 3:
            assert (
                EvidenceReply.model_validate_json(
                    ToolReply.model_validate_json(request.messages[-1].content).tool_results[0]
                ).parts
                == parts
            )
            partial: Final = invalid.model_copy(
                update={
                    "observations": (
                        invalid.observations[0].model_copy(
                            update={
                                "evidence": (expected.observations[0].evidence[0], invalid.observations[0].evidence[1])
                            }
                        ),
                    )
                }
            )
            return ModelResult(content=AgentTurn[Extraction](result=partial).model_dump_json(), cost=0)
        if turn == 4:
            assert "result.observations[0].evidence[0]" not in request.messages[-1].content
            assert "result.observations[0].evidence[1]" in request.messages[-1].content
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="history", turn_end=3),)).model_dump_json(),
                cost=0,
            )
        history: Final = JournalReply.model_validate_json(
            ToolReply.model_validate_json(request.messages[-1].content).tool_results[0]
        )
        assert len(history.turns) == 3
        assert history.turns[0].response == AgentTurn[Extraction](result=invalid).model_dump_json()
        assert "evidence[0]" in history.turns[0].validation_error
        assert "evidence[1]" in history.turns[2].validation_error
        return ModelResult(content=AgentTurn[Extraction](result=expected).model_dump_json(), cost=0)

    result: Final = await review_context(
        Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        session,
        EvidenceWorkspace(sessions=(session,)),
        model,
    )
    assert result.observations == expected.observations
    assert next(turns, None) is None


@pytest.mark.asyncio
async def test_rejected_result_remains_cancellable_without_accepting_invalid_evidence() -> None:
    session: Final = SessionContent(execution=execution("run"), parts=(), partial=False)
    correcting: Final = asyncio.Event()
    pending: Final = asyncio.Event()

    async def model(request: ModelRequest) -> ModelResult:
        if "validation_errors" in request.messages[-1].content:
            correcting.set()
            await pending.wait()
        return ModelResult(
            content=AgentTurn[Extraction](
                result=Extraction(
                    observations=(
                        Observation(
                            check_id="retries",
                            summary="Unsupported",
                            evidence=(Evidence(execution_id="run", span_id="absent", quote="invented"),),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    task: Final = asyncio.create_task(
        review_context(
            Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
            session,
            EvidenceWorkspace(sessions=(session,)),
            model,
        )
    )
    try:
        await correcting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
