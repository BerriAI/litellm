import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
from lens.agent_review import Findings
from lens.agent_runtime import PythonAgentTurn
from lens.agent_workspace import EvidenceRequest, PythonRequest
from lens.analysis import Candidate, Clusters, Extraction, Observation
from lens.models import (
    AgentTestCase,
    Check,
    Claim,
    Evidence,
    Execution,
    ExecutionContent,
    FindingDraft,
    IssueBrief,
    Job,
    LensSettings,
    ModelRequest,
    ModelResult,
    Progress,
    Result,
    Sample,
    ToolCount,
    TracePart,
)
from lens.worker import LensWorker
from pydantic import BaseModel, ConfigDict


class ToolReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tool_results: tuple[str, ...]


class PythonOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stdout: str
    exit_code: int
    output_complete: bool


class PythonReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    output: PythonOutput


async def investigate(damaged_peer: bool) -> None:
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    settings: Final = LensSettings(
        name="Tool review",
        model="stubbed-at-network-boundary",
        checks=(Check(id="tools", instruction="Find tool defects"),),
    )
    claim: Final = Claim(
        lens_id="lens",
        job=Job(id="job", created_at=now, start=now, end=now, settings=settings, revision=1),
        findings=(),
    )
    execution: Final = Execution(
        id="original-run",
        source="traces",
        trace_id="trace",
        team_id="",
        name="Task",
        start_time="",
        span_count=2,
        root_seen=True,
    )
    damaged: Final = Execution(
        id="damaged-run",
        source="traces",
        trace_id="damaged-trace",
        team_id="",
        name="Damaged source",
        start_time="",
        span_count=2,
        root_seen=True,
    )
    quote: Final = "grep: unknown option --pattern"
    nested: Final = TracePart(
        execution_id=execution.id, span_id="child", parent_span_id="root", name="grep", kind="tool", content=quote
    )
    root: Final = TracePart(
        execution_id=execution.id, span_id="root", name="Coordinator", kind="agent", content="Find matching lines"
    )
    evidence: Final = Evidence(execution_id="r0", span_id="child", quote=quote)
    finding: Final = FindingDraft(
        title="Grep argument mismatch",
        description="The nested grep call rejected its argument",
        check_id="tools",
        brief=IssueBrief(
            problem="The grep tool rejects the requested argument",
            user_goal="Find matching lines",
            what_happened=quote,
            test_cases=(AgentTestCase(input="Search for matching lines", expected="Use supported grep arguments"),),
        ),
        evidence=(evidence,),
    )
    events: Final = SimpleQueue[Progress]()
    saved: Final = SimpleQueue[Result]()

    def model(body: ModelRequest) -> str:
        if body.purpose == "cluster":
            return Clusters(
                candidates=(
                    Candidate(
                        check_id="tools",
                        kind="issue",
                        title=finding.title,
                        hypothesis=finding.description,
                        execution_ids=("p0",),
                    ),
                )
            ).model_dump_json()
        if body.purpose == "extract":
            if "Damaged source" in body.messages[1].content:
                return PythonAgentTurn[Extraction](result=Extraction()).model_dump_json()
            if len(body.messages) == 2:
                assert quote not in body.messages[1].content
                return PythonAgentTurn[Extraction](
                    tools=(
                        PythonRequest(
                            action="python",
                            code='print(sum(p["kind"] == "tool" for s in data["sessions"] for p in s["parts"]))',
                        ),
                    )
                ).model_dump_json()
            output: Final = PythonReply.model_validate_json(
                ToolReply.model_validate_json(body.messages[-1].content).tool_results[0]
            ).output
            assert output.exit_code == 0 and output.output_complete and output.stdout == "1\n"
            return PythonAgentTurn[Extraction](
                result=Extraction(
                    reasoning="The nested grep tool rejected its argument",
                    observations=(Observation(check_id="tools", summary=finding.title, evidence=(evidence,)),),
                )
            ).model_dump_json()
        if len(body.messages) == 2:
            return PythonAgentTurn[Findings](
                tools=(EvidenceRequest(action="read", execution_id="r0", span_ids=("child",)),)
            ).model_dump_json()
        assert quote in body.messages[-1].content
        return PythonAgentTurn[Findings](result=Findings(findings=(finding,))).model_dump_json()

    def handle(request: httpx.Request) -> httpx.Response:
        path: Final = request.url.path
        if path.endswith("/claim"):
            return httpx.Response(200, json=claim.model_dump(mode="json"))
        if path.endswith("/sample"):
            return httpx.Response(
                200,
                json=Sample(
                    executions=(execution, damaged) if damaged_peer else (execution,), eligible=2 if damaged_peer else 1
                ).model_dump(),
            )
        if path.endswith("/reviews"):
            return httpx.Response(200, json=[])
        if path.endswith("/content"):
            if request.url.params["execution_id"] == damaged.id:
                return httpx.Response(
                    200, json=ExecutionContent(execution=damaged, parts=(), next_cursor="repeat").model_dump()
                )
            assert request.url.params["execution_id"] == execution.id
            return httpx.Response(200, json=ExecutionContent(execution=execution, parts=(root, nested)).model_dump())
        if path.endswith("/model"):
            return httpx.Response(
                200,
                json=ModelResult(content=model(ModelRequest.model_validate_json(request.content)), cost=0).model_dump(),
            )
        if path.endswith("/result"):
            saved.put(Result.model_validate_json(request.content))
        elif path.endswith("/progress"):
            events.put(Progress.model_validate_json(request.content))
        else:
            assert path.endswith("/heartbeat"), path
        return httpx.Response(200, json=True)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client).run_once()
    result: Final = saved.get_nowait()
    assert result.coverage.unassessable == int(damaged_peer), result
    assert bool(result.error) is damaged_peer, result.error
    assert not damaged_peer or "damaged-trace" in result.error, result.error
    assert result.coverage.screened == (2 if damaged_peer else 1) and result.coverage.investigated == 1
    assert result.coverage.partial == int(damaged_peer) and result.coverage.failed_tasks == int(damaged_peer)
    expected: Final = finding.model_copy(
        update={"evidence": (evidence.model_copy(update={"execution_id": execution.id}),)}
    )
    assert result.findings == (expected,)
    progress: Final = tuple(events.get_nowait() for _ in range(events.qsize()))
    reviews: Final = tuple(event.review for event in progress if event.review is not None)
    assert len(reviews) == (2 if damaged_peer else 1)
    original_review: Final = next(review for review in reviews if review.execution_id == execution.id)
    assert original_review.tool_calls == (ToolCount(name="python", calls=1),)
    assert tuple(version.execution_id for version in result.review_versions) == (execution.id,)
    assert original_review.extraction is not None and original_review.content_version
    assert not tuple(Path("/tmp").glob("lens-python-*")), "Investigation leaked scratch"
    assert not Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip()
    assert any(event.activity is not None and "python" in event.activity.operations for event in progress)
    assert all(quote not in event.activity.model_dump_json() for event in progress if event.activity is not None)

    def reuse_handle(request: httpx.Request) -> httpx.Response:
        assert not request.url.path.endswith("/model"), "Unchanged trace called the model again"
        if request.url.path.endswith("/reviews"):
            return httpx.Response(
                200, json=[original_review.model_copy(update={"consolidated": True}).model_dump(mode="json")]
            )
        return handle(request)

    if not damaged_peer:
        async with httpx.AsyncClient(
            base_url="https://proxy.test", transport=httpx.MockTransport(reuse_handle)
        ) as client:
            assert await LensWorker(client).run_once()
        reused: Final = saved.get_nowait()
        assert reused.coverage.reused == 1 and reused.coverage.screened == 1
        assert reused.findings == () and reused.error == ""
        assert reused.review_versions == result.review_versions
    logging.warning(
        "Default worker: confined Python, live activity, nested evidence and unchanged final finding verified"
    )


async def main() -> None:
    await investigate(False)
    await investigate(True)


if __name__ == "__main__":
    asyncio.run(main())
