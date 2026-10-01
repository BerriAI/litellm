from queue import SimpleQueue
from typing import Final

import httpx
import pytest

from litellm.proxy.engine.models import Claim, Execution, ExecutionContent, ModelResult, Result, Sample, TracePart
from litellm.proxy.engine.state import queue_job
from litellm.proxy.engine.worker import EngineWorker
from tests.unit.proxy.engine.test_state import NOW, engine


@pytest.mark.asyncio
async def test_idle_worker_does_not_start_an_analysis() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/engine/worker/claim"
        return httpx.Response(200, content="null")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await EngineWorker(client).run_once() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("model_status", (200, 402, 503))
async def test_worker_reads_claimed_activity_and_reports_analysis_or_failure(model_status: int) -> None:
    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
    execution: Final = Execution(
        id="run", source="traces", trace_id="trace", team_id="alpha", name="review", start_time="", span_count=1
    )
    sample: Final = Sample(executions=(execution,), eligible=1)
    content: Final = ExecutionContent(
        execution=execution,
        parts=(TracePart(execution_id="run", span_id="span", name="lead", kind="agent", content="Completed"),),
    )
    saved: Final = SimpleQueue[Result]()

    def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path:
            case "/engine/worker/claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "/engine/worker/engine/job/sample":
                return httpx.Response(200, json=sample.model_dump(mode="json"))
            case "/engine/worker/engine/job/content":
                assert request.url.params["execution_id"] == execution.id
                return httpx.Response(200, json=content.model_dump(mode="json"))
            case "/engine/worker/engine/job/model":
                return httpx.Response(
                    model_status,
                    json=ModelResult(content='{"observations":[],"cannot_assess":false}', cost=0.01).model_dump(),
                )
            case "/engine/worker/engine/job/progress":
                return httpx.Response(200, json=True)
            case "/engine/worker/engine/job/result":
                saved.put(Result.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected analyzer request: {request.url.path}")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await EngineWorker(client).run_once() is True
    result: Final = saved.get_nowait()
    assert saved.empty()
    if model_status == 200:
        assert result.error == ""
        assert result.coverage.screened == 1
        assert result.coverage.unassessable == 0
    elif model_status == 402:
        assert result.error == "Monthly budget reached"
    else:
        assert result.error.startswith("Analysis interrupted.")
