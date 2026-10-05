import json
from queue import SimpleQueue
from typing import Final

import pytest

from litellm.proxy.lens.agent_runtime import AgentTurn, run_agent
from litellm.proxy.lens.agent_workspace import EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import Extraction, Observation
from litellm.proxy.lens.models import Claim, Evidence, ModelRequest, ModelResult, TracePart
from litellm.proxy.lens.state import queue_job
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens


@pytest.mark.asyncio
async def test_agent_reads_other_sessions_and_retains_all_prior_evidence_between_turns() -> None:
    first: Final = execution("first")
    other: Final = execution("other")
    root: Final = TracePart(execution_id=first.id, span_id="a", name="root", kind="agent", content="assigned session")
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

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        payload: Final = json.loads(request.prompt)
        assert payload["initial_evidence"][0]["content"] == root.content
        if turn == 0:
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="search", query="failure"),)
                ).model_dump_json(),
                cost=0,
            )
        previous: Final = json.loads(payload["dialogue"][0]["tool_results"][0])
        assert previous["parts"] == [nested.model_dump()]
        if turn == 1:
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="read", execution_id=other.id),)
                ).model_dump_json(),
                cost=0,
            )
        assert len(payload["dialogue"]) == 2
        assert json.loads(payload["dialogue"][1]["tool_results"][0])["parts"] == [nested.model_dump()]
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
    prompts: Final = SimpleQueue[str]()

    async def model(request: ModelRequest) -> ModelResult:
        prompts.put(request.prompt)
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
        assert workspace.respond(EvidenceRequest(action="read", execution_id=other.id)).parts == parts
        assert all(not row.spans for row in workspace.respond(EvidenceRequest(action="catalog")).catalog)
        assert len(workspace.respond(EvidenceRequest(action="catalog", execution_id=other.id)).catalog[0].spans) == len(
            parts
        )
    assert prompts.get_nowait() == prompts.get_nowait()


@pytest.mark.asyncio
async def test_checkpoint_replaces_context_but_preserves_review_records_full_history_and_valid_citations() -> None:
    from litellm.proxy.lens.agent_review import Findings, validate_findings
    from litellm.proxy.lens.agent_workspace import ReviewRecord
    from litellm.proxy.lens.models import FindingDraft
    from tests.unit.proxy.lens.test_state import issue_brief

    first: Final = execution("assigned")
    other: Final = execution("other")
    initial: Final = TracePart(
        execution_id=first.id, span_id="root", name="coordinator", kind="agent", content="Original initial material"
    )
    evidence: Final = TracePart(
        execution_id=other.id,
        span_id="child",
        parent_span_id="root",
        name="child",
        kind="agent",
        content="Required operation failed with no recovery",
    )
    review: Final = ReviewRecord(
        execution_id=other.id, phase="initial", content="Review identifies a child operation failure"
    )
    workspace: Final = EvidenceWorkspace(
        sessions=(
            SessionContent(execution=first, parts=(initial,), partial=False),
            SessionContent(execution=other, parts=(evidence,), partial=False),
        ),
        reviews=(review,),
    )
    turns: Final = iter(range(4))
    expected: Final = Findings(
        findings=(
            FindingDraft(
                title="Required operation failed",
                description="A required child operation failed before completion",
                check_id="retries",
                brief=issue_brief("The required child operation failed"),
                evidence=(Evidence(execution_id=other.id, span_id="child", quote=evidence.content),),
            ),
        )
    )

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        payload: Final = json.loads(request.prompt)
        if turn == 0:
            return ModelResult(
                content=AgentTurn[Findings](
                    tools=(
                        EvidenceRequest(action="read_reviews", execution_id=other.id, review_phase="initial"),
                        EvidenceRequest(action="read", execution_id=other.id),
                    )
                ).model_dump_json(),
                cost=0,
            )
        if turn == 1:
            assert evidence.content in request.prompt and review.content in request.prompt
            return ModelResult(
                content=AgentTurn[Findings](
                    checkpoint="Reviewed other; investigate child; assigned still pending"
                ).model_dump_json(),
                cost=0,
            )
        if turn == 2:
            assert payload["initial_evidence"] == [] and payload["supplied"] == ""
            assert evidence.content not in request.prompt and review.content not in request.prompt
            assert payload["working_notes"] == "Reviewed other; investigate child; assigned still pending"
            assert payload["journal_turns"] == 2
            return ModelResult(
                content=AgentTurn[Findings](
                    tools=(
                        EvidenceRequest(action="history", turn_start=0, turn_end=1, include_initial=True),
                        EvidenceRequest(action="read_reviews", execution_id=other.id),
                    )
                ).model_dump_json(),
                cost=0,
            )
        archived: Final = json.loads(payload["dialogue"][-1]["tool_results"][0])
        assert archived["initial_context"] == {"evidence": [initial.model_dump()], "supplied": "Initial assignment"}
        assert json.loads(archived["turns"][0]["tool_results"][1])["parts"] == [evidence.model_dump()]
        assert json.loads(payload["dialogue"][-1]["tool_results"][1])["reviews"] == [review.model_dump()]
        return ModelResult(content=AgentTurn[Findings](result=expected).model_dump_json(), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await run_agent(
        stage="aggregate",
        task="Examine all reviews",
        purpose="investigate",
        claim=claim,
        workspace=workspace,
        model=model,
        schema=Findings,
        initial_evidence=(initial,),
        supplied="Initial assignment",
        validate=lambda findings: validate_findings(claim, workspace, findings),
    )
    assert result == expected


@pytest.mark.asyncio
async def test_repeated_full_history_reads_keep_stable_references_without_recursive_copies() -> None:
    run: Final = execution("session")
    original: Final = TracePart(
        execution_id=run.id, span_id="span", name="tool", kind="tool", content="Unique original evidence " + "x" * 1000
    )
    workspace: Final = EvidenceWorkspace(sessions=(SessionContent(execution=run, parts=(original,), partial=False),))
    turns: Final = iter(range(9))
    snapshots: Final = SimpleQueue[str]()

    async def model(request: ModelRequest) -> ModelResult:
        turn: Final = next(turns)
        payload: Final = json.loads(request.prompt)
        if turn == 0:
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        if turn == 8:
            resolved: Final = json.loads(payload["dialogue"][-1]["tool_results"][0])
            assert len(resolved["turns"]) == 1
            assert json.loads(resolved["turns"][0]["tool_results"][0])["parts"] == [original.model_dump()]
            return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)
        if turn > 1:
            result_text: Final = payload["dialogue"][-1]["tool_results"][0]
            snapshots.put(result_text)
            assert result_text.count(original.content) == 1
            history: Final = json.loads(result_text)
            for index, recorded in enumerate(history["turns"][1:], start=1):
                reference = json.loads(recorded["tool_results"][0])
                assert reference["kind"] == "history_reference"
                assert reference["request"]["turn_end"] == index
                assert reference["recorded_turns"] == index
            if turn == 7:
                earliest: Final = json.loads(history["turns"][1]["tool_results"][0])
                return ModelResult(
                    content=AgentTurn[Extraction](
                        checkpoint="Resolve the earliest history reference",
                        tools=(EvidenceRequest.model_validate(earliest["request"]),),
                    ).model_dump_json(),
                    cost=0,
                )
        return ModelResult(
            content=AgentTurn[Extraction](
                checkpoint="Original evidence is in the first journal turn; retain its reference",
                tools=(EvidenceRequest(action="history"),),
            ).model_dump_json(),
            cost=0,
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await run_agent(
        stage="review",
        task="Inspect the journal",
        purpose="extract",
        claim=claim,
        workspace=workspace,
        model=model,
        schema=Extraction,
    )
    assert result.observations == ()
    assert snapshots.qsize() == 6
