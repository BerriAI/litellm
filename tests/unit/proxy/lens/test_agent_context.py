import json
from queue import SimpleQueue
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.proxy.lens.agent_context import Checkpoint, compact_context
from litellm.proxy.lens.agent_review import Findings, validate_findings
from litellm.proxy.lens.agent_runtime import (
    AgentTurn,
    DialogueTurn,
    InitialContext,
    JournalReference,
    JournalReply,
    archived_result,
    history_reply,
    run_agent,
)
from litellm.proxy.lens.agent_workspace import EvidenceReply, EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import Extraction, Observation
from litellm.proxy.lens.models import (
    Claim,
    Evidence,
    Finding,
    FindingDraft,
    ModelMessage,
    ModelRequest,
    ModelResult,
    Record,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens


class Continuation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    working_notes: str
    journal_turns: int
    resume_history_from_turn: int
    initial_context_archived: bool


class ToolResults(Record):
    journal_turns: int
    tool_results: tuple[str, ...]


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", (False, True))
async def test_checkpoint_preserves_retrieval_and_reuse_of_prior_finding_ids(automatic: bool) -> None:
    part: Final = TracePart(execution_id="one", span_id="span", name="tool", kind="tool", content="timeout")
    evidence: Final = (Evidence(execution_id="one", span_id="span", quote="timeout"),)
    prior: Final = Finding(
        id="prior-finding-sentinel",
        title="A known transient timeout",
        description="The observed timeout is already understood",
        check_id="retries",
        kind="pattern",
        status="dismissed",
        reason="The owner already reviewed this behavior",
        evidence=evidence,
        first_seen=NOW,
        last_seen=NOW,
        revision=1,
    )
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=(prior,))
    workspace: Final = EvidenceWorkspace(
        sessions=(SessionContent(execution=execution("one"), parts=(part,), partial=False),)
    )
    resume_turn: Final = 2 if automatic else 1
    turns: Final = iter(range(resume_turn + 2))

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        assert all(prior.id not in message.content for message in request.messages if message.role == "system")
        if turn == 0:
            assert prior.id in request.messages[1].content
            return ModelResult(
                content="" if automatic else AgentTurn[Findings](checkpoint="Consult prior findings").model_dump_json(),
                context_exceeded=automatic,
                cost=0,
            )
        if automatic and turn == 1:
            assert prior.id in request.messages[1].content
            return ModelResult(content=Checkpoint(working_notes="Consult prior findings").model_dump_json(), cost=0)
        if turn == resume_turn:
            continuation: Final = TypeAdapter(dict[str, JsonValue]).validate_json(request.messages[1].content)
            assert continuation["initial_context_archived"] is True
            assert all(prior.id not in message.content for message in request.messages)
            return ModelResult(
                content=AgentTurn[Findings](
                    tools=(EvidenceRequest(action="history", include_initial=True, turn_end=0),)
                ).model_dump_json(),
                cost=0,
            )
        tool_result: Final = ToolResults.model_validate_json(request.messages[-1].content)
        history: Final = JournalReply.model_validate_json(tool_result.tool_results[0])
        assert history.initial_context is not None
        assert history.initial_context.existing_findings == (prior,)
        recovered: Final = history.initial_context.existing_findings[0]
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title=recovered.title,
                            description=recovered.description,
                            check_id=recovered.check_id,
                            kind=recovered.kind,
                            existing_finding_id=recovered.id,
                            evidence=evidence,
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    result: Final = await run_agent(
        stage="investigate",
        task="Compare recorded behavior with prior findings",
        purpose="investigate",
        claim=claim,
        workspace=workspace,
        model=model,
        schema=Findings,
        validate=lambda finding: validate_findings(claim, workspace, finding),
    )
    assert result.findings[0].existing_finding_id == prior.id
    assert next(turns, None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("later_tool_result", (False, True))
async def test_repeated_compaction_preserves_unread_history_and_archived_initial_context(
    later_tool_result: bool,
) -> None:
    previous: Final = ModelMessage(
        role="user",
        content=json.dumps(
            {
                "working_notes": "Inspect unread evidence before concluding",
                "journal_turns": 10,
                "resume_history_from_turn": 4,
                "initial_context_archived": True,
            }
        ),
    )
    later: Final = (
        ModelMessage(role="assistant", content='{"tools":[{"action":"catalog"}]}'),
        ModelMessage(role="user", content='{"journal_turns":11,"tool_results":["catalog"]}'),
    )
    request: Final = ModelRequest(
        purpose="extract",
        prompt="Review the complete evidence",
        messages=(
            ModelMessage(role="system", content="Review the complete evidence"),
            previous,
            *(later if later_tool_result else ()),
        ),
    )

    async def model(checkpoint_request: ModelRequest) -> ModelResult:
        assert previous in checkpoint_request.messages
        assert checkpoint_request.messages[0].role == "system"
        assert checkpoint_request.messages[-1].role == "system"
        assert "working_notes" in checkpoint_request.messages[-1].content
        return ModelResult(
            content=Checkpoint(working_notes="Continue investigating the recorded behavior").model_dump_json(),
            cost=0,
        )

    compacted: Final = await compact_context(request, model, 11 if later_tool_result else 10, None)
    assert compacted[0] == request.messages[0]
    assert compacted[1].role == "user"
    continuation: Final = Continuation.model_validate_json(compacted[1].content)
    assert continuation.resume_history_from_turn == 4
    assert continuation.initial_context_archived is True


@pytest.mark.asyncio
async def test_automatic_notes_remain_retrievable_after_a_later_explicit_checkpoint() -> None:
    part: Final = TracePart(
        execution_id="one", span_id="span", name="tool", kind="tool", content="original recorded evidence"
    )
    notes: Final = "An unresolved lead links session one / span to the initial assignment"
    archived: Final = SimpleQueue[str]()
    turns: Final = iter(range(5))

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 0:
            return ModelResult(content="", cost=0, context_exceeded=True)
        if turn == 1:
            return ModelResult(content=Checkpoint(working_notes=notes).model_dump_json(), cost=0)
        if turn == 2:
            compacted: Final = Continuation.model_validate_json(request.messages[1].content)
            assert compacted.working_notes == notes
            assert compacted.journal_turns == 1
            archived.put(request.messages[1].content)
            return ModelResult(
                content=AgentTurn[Extraction](checkpoint="Reread the earlier reasoning next").model_dump_json(),
                cost=0,
            )
        if turn == 3:
            assert all(notes not in message.content for message in request.messages)
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="history", turn_end=1, include_initial=True),)
                ).model_dump_json(),
                cost=0,
            )
        reply: Final = ToolResults.model_validate_json(request.messages[-1].content)
        history: Final = JournalReply.model_validate_json(reply.tool_results[0])
        assert history.total_turns == 2
        assert len(history.turns) == 1
        assert history.turns[0].response == archived.get_nowait()
        assert history.turns[0].tool_results == ()
        assert history.initial_context is not None
        assert history.initial_context.evidence == (part,)
        assert history.initial_context.supplied == "Inspect this assignment"
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    result: Final = await run_agent(
        stage="review",
        task="Review",
        purpose="extract",
        claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        workspace=EvidenceWorkspace(
            sessions=(SessionContent(execution=execution("one"), parts=(part,), partial=False),)
        ),
        model=model,
        schema=Extraction,
        initial_evidence=(part,),
        supplied="Inspect this assignment",
    )
    assert result == Extraction()


@pytest.mark.asyncio
async def test_repair_overflow_recovers_omitted_evidence_without_replaying_the_malformed_response() -> None:
    part: Final = TracePart(
        execution_id="one", span_id="nested", name="child tool", kind="tool", content="original failure sentinel"
    )
    malformed: Final = "This response omitted the required JSON contract"
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id="retries",
                summary="The child tool failed",
                evidence=(Evidence(execution_id=part.execution_id, span_id=part.span_id, quote=part.content),),
            ),
        )
    )
    turns: Final = iter(range(7))

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 0:
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        if turn == 1:
            assert part.content in request.messages[-1].content
            return ModelResult(content=malformed, cost=0)
        if turn == 2:
            assert request.messages[-2] == ModelMessage(role="assistant", content=malformed)
            assert "did not match the required response contract" in request.messages[-1].content
            return ModelResult(content="", cost=0, context_exceeded=True)
        if turn == 3:
            assert "Compact this analysis conversation" in request.messages[-1].content
            assert any(message.content == malformed for message in request.messages)
            return ModelResult(content="", cost=0, context_exceeded=True)
        if turn == 4:
            assert all(part.content not in message.content for message in request.messages)
            return ModelResult(
                content=Checkpoint(working_notes="Recover original evidence from archived turn zero").model_dump_json(),
                cost=0,
            )
        assert all(message.content != malformed for message in request.messages)
        if turn == 5:
            continuation: Final = Continuation.model_validate_json(request.messages[1].content)
            assert continuation.resume_history_from_turn == 0
            assert continuation.journal_turns == 2
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="history", turn_end=1),)).model_dump_json(),
                cost=0,
            )
        reply: Final = ToolResults.model_validate_json(request.messages[-1].content)
        history: Final = JournalReply.model_validate_json(reply.tool_results[0])
        assert history.total_turns == 2
        assert EvidenceReply.model_validate_json(history.turns[0].tool_results[0]).parts == (part,)
        return ModelResult(content=AgentTurn[Extraction](result=expected).model_dump_json(), cost=0)

    result: Final = await run_agent(
        stage="review",
        task="Review",
        purpose="extract",
        claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        workspace=EvidenceWorkspace(
            sessions=(SessionContent(execution=execution("one"), parts=(part,), partial=False),)
        ),
        model=model,
        schema=Extraction,
    )
    assert result == expected


@pytest.mark.parametrize(("bounded_start", "bounded_end"), ((True, True), (True, False), (False, True)))
def test_archived_history_excerpts_remain_exact_after_the_journal_grows(bounded_start: bool, bounded_end: bool) -> None:
    sentinel: Final = "sentinel evidence"
    initial: Final = InitialContext(evidence=(), supplied="assignment")
    journal: Final = tuple(
        DialogueTurn(response=sentinel if index == 0 else "prior turn", tool_results=()) for index in range(9)
    )
    whole_request: Final = EvidenceRequest(action="history")
    whole: Final = history_reply(whole_request, initial, journal).model_dump_json()
    start: Final = whole.index(sentinel)
    request: Final = EvidenceRequest(
        action="history",
        char_start=start if bounded_start else 0,
        char_end=start + len(sentinel) if bounded_end else None,
    )
    original: Final = history_reply(request, initial, journal)
    archived: Final = archived_result(request, original.model_dump_json(), len(journal))
    later: Final = (*journal, DialogueTurn(response="retrieve history", tool_results=(archived,)))
    recovered: Final = history_reply(EvidenceRequest(action="history", turn_start=9, turn_end=10), initial, later)
    record: Final = TypeAdapter[JournalReply | JournalReference](JournalReply | JournalReference).validate_json(
        recovered.turns[0].tool_results[0]
    )
    restored: Final = history_reply(record.request, initial, later) if isinstance(record, JournalReference) else record
    assert restored.excerpt == original.excerpt
    assert sentinel in (restored.excerpt or "")
    reference: Final = JournalReference.model_validate_json(archived_result(whole_request, whole, len(journal)))
    assert reference.request.turn_end == len(journal)
    assert reference.recorded_turns == len(journal)
