import json
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
