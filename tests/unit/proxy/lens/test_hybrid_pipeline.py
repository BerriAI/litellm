import json
from queue import SimpleQueue
from typing import Final

import pytest

from litellm.proxy.lens.agent_review import Findings
from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceRequest
from litellm.proxy.lens.analysis import Candidate, Clusters, Extraction
from litellm.proxy.lens.models import (
    Claim,
    Coverage,
    Evidence,
    ExecutionContent,
    FindingDraft,
    ModelRequest,
    ModelResult,
    Sample,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from litellm.proxy.lens.worker import analyze_sample
from tests.unit.proxy.lens.test_agent_workspace import execution
from tests.unit.proxy.lens.test_state import NOW, issue_brief, lens


@pytest.mark.asyncio
async def test_independent_discovery_can_investigate_evidence_missed_by_every_session_review() -> None:
    runs: Final = (execution("first-real-session"), execution("second-real-session"))
    parts: Final = tuple(
        TracePart(
            execution_id=run.id,
            span_id="child",
            parent_span_id="root",
            name="subagent",
            kind="agent",
            content="Recorded child failure",
        )
        for run in runs
    )
    reviewed: Final = SimpleQueue[str]()
    quote: Final = Evidence(execution_id="r1", span_id="child", quote=parts[1].content)

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        index: Final = next(i for i, run in enumerate(runs) if run.id == identity)
        return ExecutionContent(execution=runs[index], parts=(parts[index],))

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        if request.purpose == "cluster":
            return ModelResult(content=json.dumps({"candidates": payload["candidates"]}), cost=0)
        stage: Final = payload["stage"]
        if stage == "context_review":
            reviewed.put(payload["initial_evidence"][0]["execution_id"])
            return ModelResult(content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0)
        if stage == "cross_session_discovery":
            assert payload["supplied"] == ""
            if not payload["dialogue"]:
                return ModelResult(
                    content=AgentTurn[Clusters](
                        tools=(EvidenceRequest(action="read", execution_id="r1"),)
                    ).model_dump_json(),
                    cost=0,
                )
            originals: Final = json.loads(payload["dialogue"][0]["tool_results"][0])["parts"]
            assert originals[0]["content"] == parts[1].content
            return ModelResult(
                content=AgentTurn[Clusters](
                    result=Clusters(
                        candidates=(
                            Candidate(
                                check_id="retries",
                                title="Investigate the child outcome",
                                hypothesis="Possible unresolved child failure",
                                execution_ids=("r1",),
                            ),
                        )
                    )
                ).model_dump_json(),
                cost=0,
            )
        assert stage == "targeted_investigation"
        assert parts[1].content in request.prompt
        assert payload["initial_evidence"][0]["parent_span_id"] == "root"
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title="Child operation failed",
                            description="The recorded child operation failed before completion",
                            check_id="retries",
                            evidence=(quote,),
                            brief=issue_brief("The required child operation failed"),
                        ),
                    )
                )
            ).model_dump_json(),
            cost=0,
        )

    async def progress(_stage: str, _coverage: Coverage) -> None:
        return None

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=runs, eligible=2), read, model, progress)
    assert frozenset(reviewed.get_nowait() for _ in range(reviewed.qsize())) == {"r0", "r1"}
    assert result.findings[0].evidence[0].execution_id == runs[1].id
    assert result.assessments[1].issue_checks == ("retries",)
    assert result.coverage.investigated == 1
