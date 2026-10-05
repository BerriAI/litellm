import json
from queue import SimpleQueue
from typing import Final

import pytest

from litellm.proxy.lens.agent_review import Findings, Hunch, SessionReview
from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceRequest
from litellm.proxy.lens.broadcast_pipeline import Broadcast
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
async def test_broadcast_revisits_clean_sessions_and_final_aggregator_can_read_original_child_evidence() -> None:
    runs: Final = (execution("first-real-session"), execution("initially-clean-real-session", 2))
    roots: Final = tuple(
        TracePart(execution_id=run.id, span_id="root", name="coordinator", kind="agent", content="Delivered answer")
        for run in runs
    )
    child: Final = TracePart(
        execution_id=runs[1].id,
        span_id="child",
        parent_span_id="root",
        name="subagent",
        kind="agent",
        content="Required operation failed in the child",
    )
    revisited: Final = SimpleQueue[str]()
    original_reads: Final = SimpleQueue[str]()
    quote: Final = Evidence(execution_id="r1", span_id="child", quote=child.content)
    new_hunch: Final = Hunch(
        check_id="retries", hypothesis="Different cause found in initially clean session", evidence=(quote,)
    )

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        index: Final = next(i for i, run in enumerate(runs) if run.id == identity)
        return ExecutionContent(execution=runs[index], parts=(roots[index], child) if index == 1 else (roots[index],))

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        stage: Final = payload["stage"]
        if stage == "session_review":
            identity: Final = payload["initial_evidence"][0]["execution_id"]
            return ModelResult(
                content=AgentTurn[SessionReview](
                    result=SessionReview(
                        execution_id=identity,
                        interpretation="Initially no issue seen",
                        hunches=(),
                    )
                ).model_dump_json(),
                cost=0,
            )
        if stage == "provisional_aggregation":
            if not payload["dialogue"]:
                assert payload["supplied"] == "" and payload["available_review_records"] == 2
                return ModelResult(
                    content=AgentTurn[Broadcast](
                        tools=(EvidenceRequest(action="read_reviews", review_phase="initial"),)
                    ).model_dump_json(),
                    cost=0,
                )
            initial_reviews: Final = json.loads(payload["dialogue"][0]["tool_results"][0])["reviews"]
            assert {review["execution_id"] for review in initial_reviews} == {"r0", "r1"}
            return ModelResult(
                content=AgentTurn[Broadcast](
                    result=Broadcast(
                        provisional_findings=(Hunch(check_id="retries", hypothesis="Check handoff outcomes"),),
                        instructions="Compare the handoff with every child outcome and report other causes",
                    )
                ).model_dump_json(),
                cost=0,
            )
        if stage == "session_revisit":
            identity: Final = payload["initial_evidence"][0]["execution_id"]
            assert "Compare the handoff with every child outcome" in payload["supplied"]
            revisited.put(identity)
            return ModelResult(
                content=AgentTurn[SessionReview](
                    result=SessionReview(
                        execution_id=identity,
                        interpretation="Rechecked all child outcomes",
                        hunches=(new_hunch,) if identity == "r1" else (),
                    )
                ).model_dump_json(),
                cost=0,
            )
        assert stage == "final_aggregation"
        if not payload["dialogue"]:
            assert new_hunch.hypothesis not in request.prompt
            assert payload["available_review_records"] == 4
            return ModelResult(
                content=AgentTurn[Findings](
                    tools=(EvidenceRequest(action="read_reviews", review_phase="revisited"),)
                ).model_dump_json(),
                cost=0,
            )
        if len(payload["dialogue"]) == 1:
            revised_reviews: Final = json.loads(payload["dialogue"][0]["tool_results"][0])["reviews"]
            assert {review["execution_id"] for review in revised_reviews} == {"r0", "r1"}
            assert new_hunch.hypothesis in json.dumps(revised_reviews)
            return ModelResult(
                content=AgentTurn[Findings](
                    tools=(EvidenceRequest(action="read", execution_id="r1", span_ids=("child",)),)
                ).model_dump_json(),
                cost=0,
            )
        content: Final = json.loads(payload["dialogue"][1]["tool_results"][0])["parts"][0]
        assert content["content"] == child.content and content["parent_span_id"] == "root"
        original_reads.put(content["execution_id"])
        return ModelResult(
            content=AgentTurn[Findings](
                result=Findings(
                    findings=(
                        FindingDraft(
                            title="Child operation failed",
                            description="The required child operation failed before the answer",
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
    assert frozenset(revisited.get_nowait() for _ in range(revisited.qsize())) == {"r0", "r1"}
    assert original_reads.get_nowait() == "r1"
    assert result.findings[0].evidence[0].execution_id == runs[1].id
    assert result.assessments[0].issue_checks == ()
    assert result.assessments[1].issue_checks == ("retries",)
