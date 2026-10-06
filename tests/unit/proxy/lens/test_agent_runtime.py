import asyncio
from itertools import chain
from queue import SimpleQueue
from typing import Final, Literal

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.lens.agent_runtime import (
    AgentTurn,
    DialogueTurn,
    InitialContext,
    JournalReply,
    PythonAgentTurn,
    history_reply,
    parallel_tools,
    run_agent,
)
from litellm.proxy.lens.agent_workspace import (
    EvidenceReply,
    EvidenceRequest,
    EvidenceWorkspace,
    PythonRequest,
    SessionContent,
)
from litellm.proxy.lens.analysis import AnalysisResponseError, Extraction, Observation
from litellm.proxy.lens.models import (
    Claim,
    Evidence,
    Finding,
    ModelMessage,
    ModelRequest,
    ModelResult,
    Record,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens


class InitialPrompt(Record):
    initial_evidence: tuple[TracePart, ...]
    supplied: str
    existing_findings: tuple[Finding, ...] = ()


class ToolReply(Record):
    journal_turns: int
    tool_results: tuple[str, ...]


class CheckpointPrompt(Record):
    working_notes: str
    initial_context_archived: bool


class CompactedPrompt(CheckpointPrompt):
    journal_turns: int
    resume_history_from_turn: int


class PythonError(Record):
    request: PythonRequest
    error: str


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_python", (False, True))
async def test_bare_final_response_is_repaired_with_the_complete_turn_schema_and_can_reread_evidence(
    enable_python: bool,
) -> None:
    from litellm.proxy.lens.agent_review import review_context

    part: Final = TracePart(
        execution_id="run", span_id="tool", name="tool", kind="tool", content="Original timeout evidence"
    )
    session: Final = SessionContent(execution=execution("run"), parts=(part,), partial=False)
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id="retries",
                summary="Tool timed out",
                evidence=(Evidence(execution_id="run", span_id="tool", quote=part.content),),
            ),
        ),
        reasoning="The original tool result records the timeout",
    )
    response_schema: Final = PythonAgentTurn[Extraction] if enable_python else AgentTurn[Extraction]
    turns: Final = iter(range(4))

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 1:
            assert part.content in request.messages[-1].content
            return ModelResult(content=expected.model_dump_json(), cost=0)
        if turn == 2:
            correction: Final = TypeAdapter(dict[str, JsonValue]).validate_json(request.messages[-1].content)
            assert correction["response_schema"] == response_schema.model_json_schema()
            assert part.content not in request.messages[-1].content
        if turn == 3:
            assert EvidenceReply.model_validate_json(
                ToolReply.model_validate_json(request.messages[-1].content).tool_results[0]
            ).parts == (part,)
            return ModelResult(content=response_schema(result=expected).model_dump_json(), cost=0)
        return ModelResult(
            content=response_schema(tools=(EvidenceRequest(action="read", execution_id="run"),)).model_dump_json(),
            cost=0,
        )

    result: Final = await review_context(
        Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        session,
        EvidenceWorkspace(sessions=(session,)),
        model,
        enable_python=enable_python,
    )
    assert result.observations == expected.observations
    assert result.parts == (part.model_copy(update={"truncated": True}),)
    assert next(turns, None) is None


@pytest.mark.asyncio
async def test_agent_reads_other_sessions_and_retains_all_prior_evidence_between_turns() -> None:
    first: Final = execution("first")
    other: Final = execution("other")
    root: Final = TracePart(
        execution_id=first.id, span_id="a", name="root", kind="agent", content="original root sentinel"
    )
    nested: Final = TracePart(
        execution_id=other.id, span_id="c", parent_span_id="b", name="child", kind="agent", content="failure found here"
    )
    workspace: Final = EvidenceWorkspace(
        sessions=(
            SessionContent(execution=first, parts=(root,), partial=False),
            SessionContent(execution=other, parts=(nested,), partial=False),
        )
    )
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id="retries",
                summary="Repeated action failed",
                evidence=(Evidence(execution_id=other.id, span_id=nested.span_id, quote="failure found here"),),
            ),
        )
    )
    turns: Final = iter((0, 1, 2))
    requests: Final = SimpleQueue[ModelRequest]()
    first_response: Final = AgentTurn[Extraction](
        tools=(EvidenceRequest(action="search", query="failure"),)
    ).model_dump_json(indent=2)

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        initial: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        assert initial.initial_evidence == (root,)
        assert request.messages[0] == ModelMessage(role="system", content=request.prompt)
        assert all(root.content not in message.content for message in request.messages if message.role == "system")
        assert all(nested.content not in message.content for message in request.messages if message.role == "system")
        if turn == 0:
            assert len(request.messages) == 2
            requests.put(request)
            return ModelResult(content=first_response, cost=0)
        previous: Final = requests.get_nowait()
        assert request.messages[:-2] == previous.messages
        requests.put(request)
        assert request.messages[2] == ModelMessage(role="assistant", content=first_response)
        first_reply: Final = ToolReply.model_validate_json(request.messages[3].content)
        assert EvidenceReply.model_validate_json(first_reply.tool_results[0]).parts == (nested,)
        if turn == 1:
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="read", execution_id=other.id),)
                ).model_dump_json(),
                cost=0,
            )
        last_reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
        assert last_reply.journal_turns == 2
        assert EvidenceReply.model_validate_json(last_reply.tool_results[0]).parts == (nested,)
        return ModelResult(content=AgentTurn[Extraction](result=expected).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await run_agent(
        stage="review",
        task="Review the recorded behavior",
        purpose="extract",
        claim=claim,
        workspace=workspace,
        model=model,
        schema=Extraction,
        initial_evidence=(root,),
    )
    assert result == expected


@pytest.mark.asyncio
async def test_initial_session_review_does_not_eagerly_embed_other_session_span_catalogs() -> None:
    run: Final = execution("assigned")
    other: Final = execution("other", 1000)
    root: Final = TracePart(execution_id=run.id, span_id="root", name="coordinator", kind="agent", content="Task")
    unrelated: Final = tuple(
        TracePart(execution_id=other.id, span_id=str(i), name=f"subagent {i}", kind="agent", content=f"evidence {i}")
        for i in range(1000)
    )
    prompts: Final = SimpleQueue[tuple[ModelMessage, ...]]()

    async def model(request: ModelRequest) -> ModelResult:
        prompts.put(request.messages)
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    for parts in ((unrelated[0],), unrelated):
        workspace: EvidenceWorkspace = EvidenceWorkspace(
            sessions=(
                SessionContent(execution=run, parts=(root,), partial=False),
                SessionContent(execution=other, parts=parts, partial=False),
            )
        )
        await run_agent(
            stage="review",
            task="Review this session",
            purpose="extract",
            claim=claim,
            workspace=workspace,
            model=model,
            schema=Extraction,
            initial_evidence=(root,),
        )
        assert (await workspace.respond(EvidenceRequest(action="read", execution_id=other.id))).parts == parts
        assert all(not row.spans for row in (await workspace.respond(EvidenceRequest(action="catalog"))).catalog)
        assert len(
            (await workspace.respond(EvidenceRequest(action="catalog", execution_id=other.id))).catalog[0].spans
        ) == len(parts)
    assert prompts.get_nowait() == prompts.get_nowait()


@pytest.mark.asyncio
async def test_disabled_python_rejects_python_call_before_execution_and_omits_python_schema() -> None:
    turns: Final = iter((0, 1, 2))
    repair_requests: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 0:
            assert '"PythonRequest"' not in request.messages[0].content
            return ModelResult(
                content=PythonAgentTurn[Extraction](
                    tools=(
                        PythonRequest(
                            action="python",
                            code="raise AssertionError('must not execute')",
                        ),
                    )
                ).model_dump_json(),
                cost=0,
            )
        if turn == 1:
            assert "did not match the required response contract" in request.messages[-1].content
            repair_requests.put(request)
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        assert request.messages[:-2] == repair_requests.get_nowait().messages
        assert ToolReply.model_validate_json(request.messages[-1].content).journal_turns == 1
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    result: Final = await run_agent(
        stage="review",
        task="Review",
        purpose="extract",
        claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        workspace=EvidenceWorkspace(),
        model=model,
        schema=Extraction,
    )
    assert result == Extraction()


@pytest.mark.asyncio
async def test_automatic_compaction_recovers_oversized_tool_output_and_preserves_findings() -> None:
    from litellm.proxy.lens.agent_context import Checkpoint

    sentinel: Final = "exact original evidence"
    part: Final = TracePart(
        execution_id="one",
        span_id="nested",
        parent_span_id="root",
        name="child",
        kind="tool",
        content=("large recorded result " * 2000) + sentinel,
    )
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id="retries",
                summary="Nested tool failure",
                evidence=(Evidence(execution_id="one", span_id="nested", quote=sentinel),),
            ),
        )
    )
    turns: Final = iter(range(7))
    full_reply: Final = SimpleQueue[str]()

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 0:
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        if turn == 1:
            oversized: Final = ToolReply.model_validate_json(request.messages[-1].content)
            full_reply.put(oversized.tool_results[0])
            assert sentinel in oversized.tool_results[0]
            return ModelResult(content="", cost=0, context_exceeded=True)
        if turn == 2:
            assert "Compact this analysis conversation" in request.messages[-1].content
            assert sentinel in request.messages[-2].content
            return ModelResult(content="", cost=0, context_exceeded=True)
        if turn == 3:
            assert all(sentinel not in message.content for message in request.messages)
            return ModelResult(
                content=Checkpoint(working_notes="Inspect the nested tool in session one").model_dump_json(), cost=0
            )
        if turn == 4:
            context: Final = CompactedPrompt.model_validate_json(request.messages[1].content)
            assert context.resume_history_from_turn == 0
            assert context.journal_turns == 2
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="history", turn_end=1, char_start=0, char_end=600),)
                ).model_dump_json(),
                cost=0,
            )
        if turn == 5:
            retrieved: Final = ToolReply.model_validate_json(request.messages[-1].content)
            history: Final = JournalReply.model_validate_json(retrieved.tool_results[0])
            assert history.excerpt is not None and len(history.excerpt) == 600
            assert history.characters > len(full_reply.get_nowait())
            assert history.total_turns == 2
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(
                        EvidenceRequest(
                            action="read",
                            execution_id="one",
                            span_ids=("nested",),
                            char_start=len(part.content) - len(sentinel),
                        ),
                    )
                ).model_dump_json(),
                cost=0,
            )
        reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
        assert EvidenceReply.model_validate_json(reply.tool_results[0]).parts[0].content == sentinel
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


def test_history_ranges_reconstruct_one_oversized_result_without_gaps() -> None:
    journal: Final = (DialogueTurn(response="read original", tool_results=("complete result " * 200,)),)
    initial: Final = InitialContext(evidence=(), supplied="original assignment")
    whole: Final = history_reply(EvidenceRequest(action="history", include_initial=True), initial, journal)
    serialized: Final = whole.model_dump_json()
    pieces: Final = tuple(
        history_reply(
            EvidenceRequest(action="history", include_initial=True, char_start=start, char_end=start + 97),
            initial,
            journal,
        )
        for start in range(0, len(serialized), 97)
    )
    assert "".join(piece.excerpt or "" for piece in pieces) == serialized
    assert all(piece.characters == len(serialized) for piece in pieces)
    catalog: Final = history_reply(EvidenceRequest(action="history", turn_end=0), initial, journal)
    assert catalog.turns == ()
    assert catalog.turn_characters == (len(journal[0].model_dump_json()),)


@pytest.mark.asyncio
async def test_unfit_task_fails_without_an_endless_compaction_loop() -> None:
    calls: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(request)
        assert calls.qsize() < 5
        return ModelResult(content="", cost=0, context_exceeded=True)

    with pytest.raises(AnalysisResponseError, match="task alone cannot fit"):
        await run_agent(
            stage="review",
            task="Review",
            purpose="extract",
            claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
            workspace=EvidenceWorkspace(),
            model=model,
            schema=Extraction,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", (False, True))
@pytest.mark.parametrize("between", ("none", "read", "checkpoint", "compaction"))
async def test_result_validation_allows_three_retries_without_resetting_after_other_turns(
    recover: bool, between: Literal["none", "read", "checkpoint", "compaction"]
) -> None:
    from litellm.proxy.lens.agent_context import Checkpoint

    rejected: Final = ModelResult(
        content=AgentTurn[Extraction](result=Extraction(reasoning="unsupported")).model_dump_json(), cost=0
    )
    accepted: Final = ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)
    continuation: Final = {
        "none": (),
        "read": (
            ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            ),
        ),
        "checkpoint": (
            ModelResult(content=AgentTurn[Extraction](checkpoint="Recheck the evidence").model_dump_json(), cost=0),
        ),
        "compaction": (
            ModelResult(content="", cost=0, context_exceeded=True),
            ModelResult(content=Checkpoint(working_notes="Recheck the evidence").model_dump_json(), cost=0),
        ),
    }[between]
    responses: Final = iter(
        (*chain.from_iterable((rejected, *continuation) for _ in range(3)), accepted if recover else rejected, accepted)
    )
    calls: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(request)
        return next(responses)

    async def run() -> Extraction:
        return await run_agent(
            stage="review",
            task="Review",
            purpose="extract",
            claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
            workspace=EvidenceWorkspace(),
            model=model,
            schema=Extraction,
            validate=lambda result: "Unsupported evidence" if result.reasoning else None,
        )

    if recover:
        assert await run() == Extraction()
    else:
        with pytest.raises(AnalysisResponseError, match="Result validation failed after 3 retries") as error:
            await run()
        assert "Unsupported evidence" in str(error.value)
    assert calls.qsize() == 4 + 3 * len(continuation)


@pytest.mark.asyncio
async def test_failed_parallel_tool_cancels_and_reaps_its_running_sibling() -> None:
    started: Final = asyncio.Event()
    stopped: Final = asyncio.Event()

    async def running() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return "unreachable"

    async def failed() -> str:
        await started.wait()
        raise ValueError("worker lease revoked")

    with pytest.raises(ValueError, match="lease revoked"):
        await parallel_tools((running(), failed()))
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_python_unknown_scope_returns_error_without_running_code() -> None:
    turns: Final = iter((0, 1))
    tool: Final = PythonRequest(
        action="python", code="raise AssertionError('must not execute')", execution_ids=("bad",)
    )

    async def model(request: ModelRequest) -> ModelResult:
        assert '"PythonRequest"' in request.messages[0].content
        if next(turns) == 0:
            return ModelResult(content=PythonAgentTurn[Extraction](tools=(tool,)).model_dump_json(), cost=0)
        reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
        assert PythonError.model_validate_json(reply.tool_results[0]) == PythonError(
            request=tool, error="Unknown execution IDs: bad"
        )
        return ModelResult(content=PythonAgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    result: Final = await run_agent(
        stage="review",
        task="Review",
        purpose="extract",
        claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        workspace=EvidenceWorkspace(),
        model=model,
        schema=Extraction,
        enable_python=True,
    )
    assert result == Extraction()


@pytest.mark.asyncio
async def test_checkpoint_replaces_active_context_and_history_preserves_original_evidence() -> None:
    part: Final = TracePart(
        execution_id="one", span_id="span", name="tool", kind="tool", content="archived checkpoint evidence sentinel"
    )
    workspace: Final = EvidenceWorkspace(
        sessions=(SessionContent(execution=execution("one"), parts=(part,), partial=False),)
    )
    turns: Final = iter(range(4))
    initial_request: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        if turn == 0:
            initial_request.put(request)
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        if turn == 1:
            reply: Final = ToolReply.model_validate_json(request.messages[-1].content)
            assert EvidenceReply.model_validate_json(reply.tool_results[0]).parts == (part,)
            return ModelResult(
                content=AgentTurn[Extraction](checkpoint="keep exact span reference").model_dump_json(), cost=0
            )
        assert (
            CheckpointPrompt.model_validate_json(request.messages[1].content).working_notes
            == "keep exact span reference"
        )
        if turn == 2:
            assert request.messages[0] == initial_request.get_nowait().messages[0]
            assert len(request.messages) == 4
            assert all(part.content not in message.content for message in request.messages)
            assert all("original instructions" not in message.content for message in request.messages)
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(
                        EvidenceRequest(
                            action="history",
                            turn_end=1,
                            include_initial=True,
                        ),
                    )
                ).model_dump_json(),
                cost=0,
            )
        history_result: Final = ToolReply.model_validate_json(request.messages[-1].content)
        history: Final = JournalReply.model_validate_json(history_result.tool_results[0])
        assert history.initial_context is not None
        assert history.initial_context.evidence == (part,)
        assert history.initial_context.supplied == "original instructions"
        assert EvidenceReply.model_validate_json(history.turns[0].tool_results[0]).parts == (part,)
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    result: Final = await run_agent(
        stage="review",
        task="Review",
        purpose="extract",
        claim=Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=()),
        workspace=workspace,
        model=model,
        schema=Extraction,
        initial_evidence=(part,),
        supplied="original instructions",
    )
    assert result == Extraction()
