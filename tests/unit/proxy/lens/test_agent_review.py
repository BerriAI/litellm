from typing import Final

import pytest

from litellm.proxy.lens.agent_review import review_context
from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceReply, EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import Extraction, Observation, review_of
from litellm.proxy.lens.models import Claim, Evidence, ModelRequest, ModelResult, TracePart
from litellm.proxy.lens.state import queue_job
from tests.unit.proxy.lens.test_agent_runtime import InitialPrompt, ToolReply
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens


@pytest.mark.parametrize("inject_evidence", (False, True))
@pytest.mark.asyncio
async def test_context_review_reads_and_cites_original_evidence_with_optional_initial_injection(
    inject_evidence: bool,
) -> None:
    part: Final = TracePart(
        execution_id="run",
        span_id="child",
        parent_span_id="parent",
        name="child",
        kind="tool",
        content="unique original failure",
    )
    session: Final = SessionContent(execution=execution("run"), parts=(part,), partial=False)
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    expected: Final = Extraction(
        observations=(
            Observation(
                check_id=claim.job.settings.analysis_checks[0].id,
                summary="Recorded failure",
                evidence=(Evidence(execution_id="run", span_id="child", quote=part.content),),
            ),
        )
    )
    turns: Final = iter((0, 1))

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = InitialPrompt.model_validate_json(request.messages[1].content)
        if next(turns) == 0:
            assert any(part.content in message.content for message in request.messages) is inject_evidence
            assert payload.initial_evidence == ((part,) if inject_evidence else ())
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
    assert result.parts == (part,)


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
                evidence=(Evidence(execution_id=other.id, span_id=related.span_id, quote=related.content),),
            ),
        ),
    )

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content=AgentTurn[Extraction](result=result).model_dump_json(), cost=0)

    examined: Final = await review_context(claim, session, workspace, model)
    review: Final = review_of(examined, claim.job.settings.model, 0, NOW)
    assert examined.parts == (root, related)
    assert examined.observations == result.observations
    assert review.execution_id == assigned.id and review.trace_id == assigned.trace_id
    assert tuple(span.span_id for span in review.spans) == (root.span_id,)
    assert review.reasoning == result.reasoning
