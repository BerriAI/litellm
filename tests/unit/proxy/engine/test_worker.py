from queue import SimpleQueue
from typing import Final

import httpx
import pytest

from litellm.proxy.engine.models import (
    Claim,
    Execution,
    ExecutionContent,
    ModelRequest,
    ModelResult,
    Result,
    Sample,
    TracePart,
)
from litellm.proxy.engine.state import queue_job
from litellm.proxy.engine.worker import EngineWorker
from tests.unit.proxy.engine.test_state import NOW, engine


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (429, 502, 503, 504, "timeout", 402, 409, 401))
async def test_model_retries_transient_failures_but_not_budget_or_revocation(failure: int | str) -> None:
    attempts: Final = SimpleQueue[str]()
    delays: Final = SimpleQueue[float]()
    expected: Final = ModelResult(content='{"observations":[]}', cost=0.01)

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.put(request.url.path)
        if attempts.qsize() == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("upstream timeout", request=request)
            assert isinstance(failure, int)
            return httpx.Response(failure)
        return httpx.Response(200, json=expected.model_dump())

    async def sleep(delay: float) -> None:
        delays.put(delay)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        worker: Final = EngineWorker(client, sleep=sleep)
        if failure in (402, 409, 401):
            with pytest.raises(httpx.HTTPStatusError):
                await worker.model_request("/model", ModelRequest(purpose="extract", prompt="review"))
            assert attempts.qsize() == 1 and delays.empty()
        else:
            assert await worker.model_request("/model", ModelRequest(purpose="extract", prompt="review")) == expected
            assert attempts.qsize() == 2
            assert delays.get_nowait() == 1 and delays.empty()


@pytest.mark.asyncio
async def test_transient_retries_are_bounded() -> None:
    attempts: Final = SimpleQueue[str]()
    delays: Final = SimpleQueue[float]()

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.put(request.url.path)
        return httpx.Response(503)

    async def sleep(delay: float) -> None:
        delays.put(delay)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await EngineWorker(client, sleep=sleep).model_request(
                "/model", ModelRequest(purpose="extract", prompt="review")
            )
    assert attempts.qsize() == 3
    assert tuple(delays.get_nowait() for _ in range(delays.qsize())) == (1, 2)


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
