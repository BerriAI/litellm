import json
from queue import SimpleQueue
from typing import Final

import pytest

from litellm.proxy.lens.agent_runtime import AgentTurn, PythonAgentTurn, run_agent
from litellm.proxy.lens.agent_workspace import (
    EvidenceReply,
    EvidenceRequest,
    EvidenceWorkspace,
    PythonRequest,
    SearchMatch,
    SessionContent,
)
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
        previous: Final = EvidenceReply.model_validate_json(payload["dialogue"][0]["tool_results"][0])
        assert previous.parts == ()
        assert previous.matches == (
            SearchMatch(
                execution_id=other.id,
                span_id="c",
                parent_span_id="b",
                name="child",
                kind="agent",
                char_start=0,
                char_end=7,
                characters=len(nested.content),
            ),
        )
        if turn == 1:
            hit: Final = previous.matches[0]
            return ModelResult(
                content=AgentTurn[Extraction](
                    tools=(EvidenceRequest(action="read", execution_id=hit.execution_id, span_ids=(hit.span_id,)),)
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
async def test_disabled_python_rejects_python_call_before_execution_and_omits_python_schema() -> None:
    turns: Final = iter((0, 1))

    async def model(request: ModelRequest) -> ModelResult:
        if next(turns) == 0:
            assert "PythonRequest" not in json.loads(request.prompt)["response_schema"].get("$defs", {})
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
        assert "did not match the required response contract" in request.prompt
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
async def test_python_unknown_scope_returns_error_without_running_code() -> None:
    turns: Final = iter((0, 1))
    tool: Final = PythonRequest(
        action="python", code="raise AssertionError('must not execute')", execution_ids=("bad",)
    )

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        assert "PythonRequest" in payload["response_schema"]["$defs"]
        if next(turns) == 0:
            return ModelResult(content=PythonAgentTurn[Extraction](tools=(tool,)).model_dump_json(), cost=0)
        assert json.loads(payload["dialogue"][0]["tool_results"][0]) == {
            "request": tool.model_dump(mode="json"),
            "error": "Unknown execution IDs: bad",
        }
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
    part: Final = TracePart(execution_id="one", span_id="span", name="tool", kind="tool", content="original evidence")
    workspace: Final = EvidenceWorkspace(
        sessions=(SessionContent(execution=execution("one"), parts=(part,), partial=False),)
    )
    turns: Final = iter(range(4))

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        turn: Final = next(turns)
        if turn == 0:
            return ModelResult(
                content=AgentTurn[Extraction](tools=(EvidenceRequest(action="read"),)).model_dump_json(), cost=0
            )
        if turn == 1:
            assert json.loads(payload["dialogue"][0]["tool_results"][0])["parts"][0]["content"] == part.content
            return ModelResult(
                content=AgentTurn[Extraction](checkpoint="keep exact span reference").model_dump_json(), cost=0
            )
        assert payload["initial_evidence"] == [] and payload["supplied"] == ""
        assert payload["working_notes"] == "keep exact span reference"
        if turn == 2:
            assert len(payload["dialogue"]) == 1
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
        history: Final = json.loads(payload["dialogue"][-1]["tool_results"][0])
        assert history["initial_context"] == {
            "evidence": [part.model_dump(mode="json")],
            "supplied": "original instructions",
        }
        assert json.loads(history["turns"][0]["tool_results"][0])["parts"] == [part.model_dump(mode="json")]
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
