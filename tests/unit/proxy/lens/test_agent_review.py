import json
from typing import Final

import pytest

from litellm.proxy.lens.agent_review import review_context
from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceRequest, EvidenceWorkspace, SessionContent
from litellm.proxy.lens.analysis import Extraction, Observation
from litellm.proxy.lens.models import Claim, Evidence, ModelRequest, ModelResult, TracePart
from litellm.proxy.lens.state import queue_job
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
        payload: Final = json.loads(request.prompt)
        if next(turns) == 0:
            assert (part.content in request.prompt) is inject_evidence
            assert payload["initial_evidence"] == ([part.model_dump(mode="json")] if inject_evidence else [])
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
        assert json.loads(payload["dialogue"][0]["tool_results"][0])["parts"] == [part.model_dump(mode="json")]
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
