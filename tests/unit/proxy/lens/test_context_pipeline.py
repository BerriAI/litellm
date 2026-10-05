from typing import Final

import pytest

from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.analysis import Extraction
from litellm.proxy.lens.models import Claim, Coverage, ExecutionContent, ModelRequest, ModelResult, Sample, TracePart
from litellm.proxy.lens.state import queue_job
from litellm.proxy.lens.worker import analyze_sample
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, lens


@pytest.mark.asyncio
async def test_production_entrypoint_reads_full_child_content_before_reviewing_the_session() -> None:
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
        assert child.content in request.prompt
        assert '"parent_span_id": "root"' in request.prompt
        return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)

    async def progress(_stage: str, _coverage: Coverage) -> None:
        return None

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=(run,), eligible=1), read, model, progress)
    assert result.coverage.screened == 1
    assert result.assessments[0].execution_id == run.id
    assert result.findings == ()
