import asyncio
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from litellm.proxy.lens.agent_runtime import AgentTurn
from litellm.proxy.lens.agent_workspace import EvidenceRequest
from litellm.proxy.lens.analysis import Extraction, analyze_sample
from litellm.proxy.lens.models import (
    Claim,
    Execution,
    ExecutionContent,
    ModelMessage,
    ModelRequest,
    ModelResult,
    Progress,
    Result,
    Review,
    Sample,
    ToolCount,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from litellm.proxy.lens.worker import (
    MODEL_RETRIES,
    MODEL_RETRY_MAX_SECONDS,
    LensWorker,
    failure_message,
    retry_delay,
)
from tests.unit.proxy.lens.test_state import NOW, lens


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (429, 502, 503, 504, "timeout", 402, 409, 401))
async def test_model_retries_transient_failures_but_not_budget_or_revocation(failure: int | str) -> None:
    attempts: Final = SimpleQueue[str]()
    delays: Final = SimpleQueue[float]()
    expected: Final = ModelResult(content='{"observations":[]}', cost=0.01)
    body: Final = ModelRequest(
        purpose="extract",
        prompt="review",
        messages=(
            ModelMessage(role="user", content="review"),
            ModelMessage(role="assistant", content='{ "tools": [{"action": "read"}] }'),
            ModelMessage(role="user", content="Full original evidence"),
        ),
    )

    def handle(request: httpx.Request) -> httpx.Response:
        assert ModelRequest.model_validate_json(request.content) == body
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
        worker: Final = LensWorker(client, analysis=analyze_sample, sleep=sleep)
        if failure in (402, 409, 401):
            with pytest.raises(httpx.HTTPStatusError):
                await worker.model_request("/model", body)
            assert attempts.qsize() == 1 and delays.empty()
        else:
            assert await worker.model_request("/model", body) == expected
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
            await LensWorker(client, analysis=analyze_sample, sleep=sleep).model_request(
                "/model", ModelRequest(purpose="extract", prompt="review")
            )
    assert attempts.qsize() == MODEL_RETRIES + 1
    assert tuple(delays.get_nowait() for _ in range(delays.qsize())) == tuple(
        float(min(2**n, MODEL_RETRY_MAX_SECONDS)) for n in range(MODEL_RETRIES)
    )


@pytest.mark.asyncio
async def test_rate_limited_model_waits_as_long_as_the_provider_asks_then_completes() -> None:
    attempts: Final = SimpleQueue[str]()
    delays: Final = SimpleQueue[float]()
    expected: Final = ModelResult(content='{"observations":[]}', cost=0.01)

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.put(request.url.path)
        if attempts.qsize() <= 3:
            return httpx.Response(429, headers={"retry-after": "30"})
        return httpx.Response(200, json=expected.model_dump())

    async def sleep(delay: float) -> None:
        delays.put(delay)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        result: Final = await LensWorker(client, sleep=sleep).model_request(
            "/model", ModelRequest(purpose="extract", prompt="review")
        )
    assert result == expected
    assert tuple(delays.get_nowait() for _ in range(delays.qsize())) == (30, 30, 30)


@pytest.mark.parametrize(
    ("retry_after", "attempt", "expected"),
    (("", 1, 2), ("5", 0, 5), ("1", 3, 8), ("9999", 0, MODEL_RETRY_MAX_SECONDS), ("soon", 2, 4)),
)
def test_retry_delay_prefers_the_providers_wait_within_bounds(retry_after: str, attempt: int, expected: float) -> None:
    request: Final = httpx.Request("POST", "https://proxy.test/model")
    headers: Final = {"retry-after": retry_after} if retry_after else {}
    error: Final = httpx.HTTPStatusError("limited", request=request, response=httpx.Response(429, headers=headers))
    assert retry_delay(error, attempt) == expected


@pytest.mark.asyncio
async def test_idle_worker_does_not_start_an_analysis() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/lens/worker/claim"
        return httpx.Response(200, content="null")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client, analysis=analyze_sample).run_once() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("result_status", (200, 409))
async def test_incompatible_claim_reports_failure_instead_of_leaving_the_investigation_running(
    result_status: int,
) -> None:
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    payload: Final = claim.model_dump(mode="json") | {
        "job": claim.job.model_dump(mode="json")
        | {
            "settings": claim.job.settings.model_dump() | {"future_setting": "private content"},
        },
    }
    saved: Final = SimpleQueue[Result]()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/lens/worker/claim":
            return httpx.Response(200, json=payload)
        assert request.url.path == "/lens/worker/lens/job/result"
        saved.put(Result.model_validate_json(request.content))
        return httpx.Response(result_status, json=True)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client, analysis=analyze_sample).run_once() is True
    assert saved.get_nowait().error == (
        "The worker could not read this investigation. Update the worker to match the gateway, then retry."
    )
    assert saved.empty()


@pytest.mark.asyncio
async def test_claim_without_an_identity_does_not_report_failure_for_another_investigation() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/lens/worker/claim"
        return httpx.Response(200, json={"job": {"settings": {"future_setting": True}}})

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValidationError):
            await LensWorker(client, analysis=analyze_sample).run_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_status", (200, 402, 503))
async def test_worker_reads_claimed_activity_and_reports_analysis_or_failure(model_status: int) -> None:
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
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
            case "/lens/worker/claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "/lens/worker/lens/job/reviews":
                return httpx.Response(200, json=[])
            case "/lens/worker/lens/job/sample":
                return httpx.Response(200, json=sample.model_dump(mode="json"))
            case "/lens/worker/lens/job/content":
                assert request.url.params["execution_id"] == execution.id
                return httpx.Response(200, json=content.model_dump(mode="json"))
            case "/lens/worker/lens/job/model":
                return httpx.Response(
                    model_status,
                    json=ModelResult(content='{"observations":[],"cannot_assess":false}', cost=0.01).model_dump(),
                )
            case "/lens/worker/lens/job/progress":
                return httpx.Response(200, json=True)
            case "/lens/worker/lens/job/result":
                saved.put(Result.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected analyzer request: {request.url.path}")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client, analysis=analyze_sample).run_once() is True
    result: Final = saved.get_nowait()
    assert saved.empty()
    if model_status == 200:
        assert result.error == ""
        assert result.coverage.screened == 1
        assert result.coverage.unassessable == 0
    elif model_status == 402:
        assert "HTTP 402" in result.error and "remaining budget" in result.error
    else:
        assert result.error.startswith("Model request failed (HTTP 503).")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (401, 402, 409, 503, "timeout"))
async def test_model_failure_stops_remaining_traces_without_discarding_completed_reviews(failure: int | str) -> None:
    initial: Final = lens()
    configured: Final = initial.model_copy(update={"settings": initial.settings.model_copy(update={"concurrency": 1})})
    claim: Final = Claim(lens_id="lens", job=queue_job(configured, NOW, "job").jobs[0], findings=())
    executions: Final = tuple(
        Execution(
            id=identity,
            source="traces",
            trace_id=identity,
            team_id="alpha",
            name="review",
            start_time="",
            span_count=1,
            root_seen=True,
        )
        for identity in ("healthy", "blocked", "unstarted")
    )
    requests: Final = SimpleQueue[str]()
    checkpoints: Final = SimpleQueue[Progress]()
    results: Final = SimpleQueue[Result]()

    def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path.rsplit("/", 1)[-1]:
            case "claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "reviews":
                return httpx.Response(200, json=[])
            case "sample":
                return httpx.Response(200, json=Sample(executions=executions, eligible=3).model_dump(mode="json"))
            case "content":
                identity: Final = request.url.params["execution_id"]
                content: Final = ExecutionContent(
                    execution=next(execution for execution in executions if execution.id == identity),
                    parts=(TracePart(execution_id=identity, span_id="span", name="tool", kind="tool", content="done"),),
                )
                return httpx.Response(200, json=content.model_dump(mode="json"))
            case "model":
                requests.put(request.url.path)
                if requests.qsize() == 1:
                    return httpx.Response(
                        200,
                        json=ModelResult(
                            content=AgentTurn[Extraction](result=Extraction()).model_dump_json(), cost=0.01
                        ).model_dump(),
                    )
                if failure == "timeout":
                    raise httpx.ReadTimeout("private provider diagnostics", request=request)
                return httpx.Response(int(failure))
            case "progress":
                progress: Final = Progress.model_validate_json(request.content)
                if progress.review is not None:
                    checkpoints.put(progress)
                return httpx.Response(200, json=True)
            case "result":
                results.put(Result.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected worker request: {request.url.path}")

    async def no_delay(_seconds: float) -> None:
        return None

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client, sleep=no_delay).run_once()
    saved: Final = checkpoints.get_nowait()
    assert saved.review is not None and saved.review.execution_id == "healthy"
    assert saved.review.extraction is not None and saved.review.content_version
    assert checkpoints.empty()
    stopped: Final = results.get_nowait()
    assert stopped.error and stopped.findings == ()
    assert tuple((item.execution_id, item.cannot_assess) for item in stopped.assessments) == (("healthy", False),)
    assert stopped.coverage.screened == 1 and stopped.coverage.unassessable == 0
    assert stopped.review_versions == ()
    assert "private provider diagnostics" not in stopped.error
    assert requests.qsize() == 2 + (MODEL_RETRIES if failure in (503, "timeout") else 0)
    assert results.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("cluster", "investigate", "consolidate"))
async def test_model_failure_preserves_reviews_without_publishing_unreconciled_findings(stage: str) -> None:
    from litellm.proxy.lens.agent_review import Findings
    from litellm.proxy.lens.analysis import Candidate, Clusters
    from litellm.proxy.lens.endpoints import merge_results
    from litellm.proxy.lens.models import Evidence, FindingDraft, Observation
    from litellm.proxy.lens.reconciliation import FindingGroup, FindingGroups
    from litellm.proxy.lens.state import merge_finding
    from tests.unit.proxy.lens.test_context_pipeline import AssignedSession, GroupPrompt
    from tests.unit.proxy.lens.test_state import finding, issue_brief

    class SuppliedPrompt(BaseModel):
        supplied: str

    initial: Final = lens()
    prior: Final = merge_finding(initial, finding("earlier"), 1, NOW)
    configured: Final = initial.model_copy(
        update={"settings": initial.settings.model_copy(update={"concurrency": 1}), "findings": (prior,)}
    )
    claim: Final = Claim(lens_id="lens", job=queue_job(configured, NOW, "job").jobs[0], findings=(prior,))
    executions: Final = tuple(
        Execution(
            id=identity,
            source="traces",
            trace_id=identity,
            team_id="",
            name=identity,
            start_time="",
            span_count=1,
            root_seen=True,
        )
        for identity in ("first", "second")
    )
    failed: Final = asyncio.Event()
    resuming: Final = asyncio.Event()
    investigated: Final = SimpleQueue[str]()
    saved: Final = SimpleQueue[Result]()
    checkpoints: Final = SimpleQueue[Review]()

    def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path.rsplit("/", 1)[-1]:
            case "claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "reviews":
                return httpx.Response(
                    200, json=[review.model_dump(mode="json") for review in retained] if resuming.is_set() else []
                )
            case "sample":
                return httpx.Response(200, json=Sample(executions=executions, eligible=2).model_dump(mode="json"))
            case "content":
                identity: Final = request.url.params["execution_id"]
                return httpx.Response(
                    200,
                    json=ExecutionContent(
                        execution=next(item for item in executions if item.id == identity),
                        parts=(
                            TracePart(
                                execution_id=identity, span_id="span", name="tool", kind="tool", content="timeout"
                            ),
                        ),
                    ).model_dump(mode="json"),
                )
            case "model":
                body: Final = ModelRequest.model_validate_json(request.content)
                if resuming.is_set():
                    assert body.purpose != "extract", "A retry must reuse completed trace reviews"
                else:
                    assert not failed.is_set(), "A terminal model error must stop further model calls"
                consolidation: Final = '"FindingGroups"' in body.prompt
                if not resuming.is_set() and (
                    (stage == "consolidate" and consolidation)
                    or (stage == body.purpose and (stage != "investigate" or investigated.qsize() == 1))
                ):
                    failed.set()
                    return httpx.Response(402, text="private provider diagnostics")
                if consolidation:
                    return httpx.Response(
                        200,
                        json=ModelResult(
                            content=FindingGroups(
                                groups=(
                                    FindingGroup(
                                        members=("new:0", "new:1", f"saved:{prior.id}"),
                                        representative=f"saved:{prior.id}",
                                    ),
                                )
                            ).model_dump_json(),
                            cost=0.01,
                        ).model_dump(),
                    )
                if body.purpose == "cluster":
                    groups: Final = GroupPrompt.model_validate_json(body.prompt)
                    return httpx.Response(
                        200,
                        json=ModelResult(
                            content=Clusters(candidates=groups.candidates).model_dump_json(),
                            cost=0.01,
                        ).model_dump(),
                    )
                payload: Final = SuppliedPrompt.model_validate_json(body.messages[1].content)
                if body.purpose == "extract":
                    assigned: Final = AssignedSession.model_validate_json(payload.supplied).execution
                    return httpx.Response(
                        200,
                        json=ModelResult(
                            content=AgentTurn[Extraction](
                                result=Extraction(
                                    observations=(
                                        Observation(
                                            check_id="retries",
                                            summary=assigned.name,
                                            evidence=(
                                                Evidence(execution_id=assigned.id, span_id="span", quote="timeout"),
                                            ),
                                        ),
                                    ),
                                )
                            ).model_dump_json(),
                            cost=0.01,
                        ).model_dump(),
                    )
                candidate: Final = Candidate.model_validate_json(payload.supplied)
                investigated.put(candidate.title)
                return httpx.Response(
                    200,
                    json=ModelResult(
                        content=AgentTurn[Findings](
                            result=Findings(
                                findings=(
                                    FindingDraft(
                                        title=candidate.title,
                                        description="A recorded operation timed out",
                                        check_id="retries",
                                        brief=issue_brief("The operation timed out"),
                                        evidence=(
                                            Evidence(
                                                execution_id=candidate.execution_ids[0], span_id="span", quote="timeout"
                                            ),
                                        ),
                                    ),
                                )
                            )
                        ).model_dump_json(),
                        cost=0.01,
                    ).model_dump(),
                )
            case "progress":
                update: Final = Progress.model_validate_json(request.content)
                if update.review is not None:
                    checkpoints.put(update.review)
                return httpx.Response(200, json=True)
            case "result":
                saved.put(Result.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected request: {request.url.path}")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client).run_once()
    result: Final = saved.get_nowait()
    assert failed.is_set() and "HTTP 402" in result.error and "private" not in result.error
    assert tuple((item.execution_id, item.issue_checks) for item in result.assessments) == (
        ("first", ("retries",)),
        ("second", ("retries",)),
    )
    assert result.findings == ()
    assert merge_results(configured, result, 1, NOW, "job").findings == (prior,)
    retained: Final = tuple(checkpoints.get_nowait() for _ in executions)
    for execution, checkpoint in zip(executions, retained):
        assert checkpoint.execution_id == execution.id and checkpoint.content_version
        assert checkpoint.extraction is not None and checkpoint.extraction.observations
        assert not checkpoint.consolidated
    assert checkpoints.empty()
    assert result.coverage.screened == 2 and result.coverage.unassessable == 0
    assert result.coverage.investigated == investigated.qsize()
    assert result.review_versions == ()
    assert saved.empty()
    resuming.set()
    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client).run_once()
    retried: Final = saved.get_nowait()
    assert not retried.error and retried.coverage.reused == 2
    assert len(retried.review_versions) == 2
    merged: Final = merge_results(configured, retried, 1, NOW, "retry").findings
    assert len(merged) == 1 and merged[0].id == prior.id
    assert frozenset(merged[0].occurrences) == frozenset(("earlier", "first", "second"))
    assert frozenset(prior.evidence) <= frozenset(merged[0].evidence)


@pytest.mark.parametrize("status", (400, 401, 402, 403, 404, 409, 429, 503))
def test_failure_reports_action_and_status_without_private_response_content(status: int) -> None:
    request: Final = httpx.Request(
        "POST", "https://private-host.test/lens/worker/private-lens/private-run/model?token=secret"
    )
    response: Final = httpx.Response(status, request=request, text="private trace content and key")
    error: Final = httpx.HTTPStatusError("private exception details", request=request, response=response)
    message: Final = failure_message(error)
    assert message.startswith(f"Model request failed (HTTP {status}).")
    assert "private" not in message and "secret" not in message


@pytest.mark.parametrize(
    "route,action", (("sample", "Reading trace data"), ("content", "Reading trace data"), ("result", "Saving results"))
)
def test_failure_identifies_the_failing_worker_operation(route: str, action: str) -> None:
    request: Final = httpx.Request("GET", f"https://proxy.test/lens/worker/lens/job/{route}")
    response: Final = httpx.Response(503, request=request)
    error: Final = httpx.HTTPStatusError("private body", request=request, response=response)
    assert failure_message(error).startswith(f"{action} failed (HTTP 503).")


def test_connection_timeout_and_invalid_response_have_distinct_private_diagnostics() -> None:
    assert "connect to the proxy" in failure_message(httpx.ConnectError("private hostname"))
    assert "timed out" in failure_message(httpx.ReadTimeout("private prompt"))
    assert "structured JSON" in failure_message(ValueError("private model response"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "purpose,stage,schema",
    (
        ("extract", "Reading executions", "TraceReview"),
        ("cluster", "Grouping observations", "Clusters"),
        ("investigate", "Checking original evidence", "Decision"),
    ),
)
async def test_worker_saves_validation_errors_from_every_analysis_stage(purpose: str, stage: str, schema: str) -> None:
    import json

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    execution: Final = Execution(
        id="run", source="traces", trace_id="trace", team_id="alpha", name="review", start_time="", span_count=1
    )
    sample: Final = Sample(executions=(execution,), eligible=1)
    content: Final = ExecutionContent(
        execution=execution,
        parts=(TracePart(execution_id="run", span_id="span", name="lead", kind="agent", content="Tool timeout"),),
    )
    saved: Final = SimpleQueue[Result]()
    attempts: Final = SimpleQueue[str]()

    def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path.rsplit("/", 1)[-1]:
            case "claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "reviews":
                return httpx.Response(200, json=[])
            case "sample":
                return httpx.Response(200, json=sample.model_dump(mode="json"))
            case "content":
                return httpx.Response(200, json=content.model_dump(mode="json"))
            case "model":
                body: Final = ModelRequest.model_validate_json(request.content)
                if body.purpose == purpose:
                    attempts.put(body.purpose)
                    return httpx.Response(
                        200,
                        json={"content": '{"candidates":[', "cost": 0.01},
                        headers={"x-litellm-lens-finish-reason": "length"},
                    )
                if body.purpose == "cluster":
                    return httpx.Response(
                        200,
                        json={
                            "content": json.dumps({"candidates": json.loads(body.prompt)["candidates"]}),
                            "cost": 0.01,
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "content": json.dumps(
                            {
                                "observations": [
                                    {
                                        "check_id": claim.job.settings.analysis_checks[0].id,
                                        "summary": "Tool timeout",
                                        "evidence": [
                                            {"execution_id": "r0", "span_id": "span", "quote": "Tool timeout"}
                                        ],
                                    }
                                ]
                            }
                        ),
                        "cost": 0.01,
                    },
                )
            case "progress":
                return httpx.Response(200, json=True)
            case "result":
                saved.put(Result.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected worker request: {request.url.path}")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client, analysis=analyze_sample).run_once()
    message: Final = saved.get_nowait().error
    assert message.startswith(f"{stage} failed: {schema} response invalid after 2 attempts.")
    assert "finish_reason=length" in message
    assert "EOF while parsing" in message and "[json_invalid]" in message
    assert attempts.qsize() == 2 and saved.empty()


def test_response_validation_diagnostics_omit_input_values_and_unexpected_field_names() -> None:
    with pytest.raises(ValidationError) as caught:
        ModelResult.model_validate({"content": "private trace", "cost": "private token", "private field": "secret"})
    message: Final = failure_message(caught.value)
    assert "Invalid ModelResult response" in message
    assert "cost:" in message and "[float_parsing]" in message
    assert "[extra_forbidden]" in message
    assert "private" not in message and "secret" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("heartbeat_status", (401, 403, 409))
async def test_losing_the_lease_interrupts_an_in_flight_model_request(heartbeat_status: int) -> None:
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    started: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()
    never: Final = asyncio.Event()
    saved: Final = SimpleQueue[Result]()

    async def heartbeat_wait(_seconds: float) -> None:
        await started.wait()

    async def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path.rsplit("/", 1)[-1]:
            case "claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "reviews":
                return httpx.Response(200, json=[])
            case "sample":
                return httpx.Response(200, json=Sample(executions=(execution,), eligible=1).model_dump())
            case "content":
                return httpx.Response(
                    200,
                    json=ExecutionContent(
                        execution=execution,
                        parts=(
                            TracePart(execution_id="run", span_id="span", name="step", kind="tool", content="evidence"),
                        ),
                    ).model_dump(),
                )
            case "model":
                assert request.extensions["timeout"] == {"connect": 13, "read": None, "write": 13, "pool": 13}
                started.set()
                try:
                    await never.wait()
                finally:
                    cancelled.set()
                pytest.fail("The cancelled model request must not finish")
            case "heartbeat":
                return httpx.Response(heartbeat_status)
            case "progress":
                return httpx.Response(200, json=True)
            case "result":
                saved.put(Result.model_validate_json(request.content))
                return httpx.Response(409)
            case _:
                pytest.fail(f"Unexpected worker request: {request.url.path}")

    async with httpx.AsyncClient(
        base_url="https://proxy.test", transport=httpx.MockTransport(handle), timeout=13
    ) as client:
        assert await LensWorker(client, analysis=analyze_sample, heartbeat_wait=heartbeat_wait).run_once()
    assert cancelled.is_set()
    assert f"HTTP {heartbeat_status}" in saved.get_nowait().error
    assert saved.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (429, 500, 502, 503, 504, "connection", "timeout"))
async def test_transient_heartbeat_failure_recovers_without_cancelling_analysis(failure: int | str) -> None:
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    started: Final = asyncio.Event()
    recovered: Final = asyncio.Event()
    never: Final = asyncio.Event()
    attempts: Final = SimpleQueue[str]()
    saved: Final = SimpleQueue[Result]()

    async def heartbeat_wait(_seconds: float) -> None:
        await started.wait()
        if attempts.qsize() >= 2:
            await never.wait()

    async def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path.rsplit("/", 1)[-1]:
            case "claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "reviews":
                return httpx.Response(200, json=[])
            case "sample":
                return httpx.Response(200, json=Sample(executions=(execution,), eligible=1).model_dump())
            case "content":
                return httpx.Response(
                    200,
                    json=ExecutionContent(
                        execution=execution,
                        parts=(
                            TracePart(execution_id="run", span_id="span", name="step", kind="tool", content="evidence"),
                        ),
                    ).model_dump(),
                )
            case "model":
                started.set()
                await recovered.wait()
                return httpx.Response(200, json={"content": '{"observations":[],"cannot_assess":false}', "cost": 0.01})
            case "heartbeat":
                attempts.put(request.url.path)
                if attempts.qsize() == 1:
                    if failure == "connection":
                        raise httpx.ConnectError("temporary connection failure", request=request)
                    if failure == "timeout":
                        raise httpx.ReadTimeout("temporary response timeout", request=request)
                    assert isinstance(failure, int)
                    return httpx.Response(failure)
                recovered.set()
                return httpx.Response(200, json=True)
            case "progress":
                return httpx.Response(200, json=True)
            case "result":
                saved.put(Result.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected worker request: {request.url.path}")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client, analysis=analyze_sample, heartbeat_wait=heartbeat_wait).run_once()
    result: Final = saved.get_nowait()
    assert result.error == ""
    assert result.coverage.screened == 1 and result.coverage.unassessable == 0
    assert attempts.qsize() == 2 and saved.empty()


@pytest.mark.asyncio
async def test_worker_sends_each_runs_review_with_its_progress() -> None:
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    sent: Final = SimpleQueue[Progress]()

    def handle(request: httpx.Request) -> httpx.Response:
        match request.url.path.rsplit("/", 1)[-1]:
            case "claim":
                return httpx.Response(200, json=claim.model_dump(mode="json"))
            case "reviews":
                return httpx.Response(200, json=[])
            case "sample":
                return httpx.Response(200, json=Sample(executions=(execution,), eligible=1).model_dump())
            case "content":
                return httpx.Response(
                    200,
                    json=ExecutionContent(
                        execution=execution,
                        parts=(TracePart(execution_id="run", span_id="s", name="step", kind="agent", content="Done"),),
                    ).model_dump(),
                )
            case "model":
                body: Final = ModelRequest.model_validate_json(request.content)
                answer: Final = (
                    AgentTurn[Extraction](tools=(EvidenceRequest(action="read", execution_id="r0"),))
                    if len(body.messages) == 2
                    else AgentTurn[Extraction](result=Extraction(reasoning="Finished the task."))
                )
                return httpx.Response(200, json={"content": answer.model_dump_json(), "cost": 0})
            case "progress":
                sent.put(Progress.model_validate_json(request.content))
                return httpx.Response(200, json=True)
            case "result":
                return httpx.Response(200, json=True)
            case _:
                pytest.fail(f"Unexpected worker request: {request.url.path}")

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert await LensWorker(client).run_once()
    reviews: Final = tuple(p.review for p in (sent.get_nowait() for _ in range(sent.qsize())) if p.review)
    assert tuple((r.execution_id, r.reasoning) for r in reviews) == (("run", "Finished the task."),)
    assert reviews[0].tool_calls == (ToolCount(name="read", calls=1),)


@pytest.mark.asyncio
async def test_worker_runs_investigations_in_parallel_and_polls_quickly_when_idle() -> None:
    claims: Final = SimpleQueue[str]()
    running: Final = asyncio.Event()
    waits: Final = SimpleQueue[float]()

    class Worker(LensWorker):
        async def run_once(self) -> bool:
            claims.put("claim")
            if claims.qsize() <= 2:
                if claims.qsize() == 2:
                    running.set()
                await running.wait()
                return True
            raise asyncio.CancelledError

    async def sleep(delay: float) -> None:
        waits.put(delay)

    async with httpx.AsyncClient(base_url="https://proxy.test") as client:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(Worker(client, sleep=sleep).serve(slots=2, poll_seconds=2), timeout=1)
    assert running.is_set()
    assert waits.empty()


@pytest.mark.asyncio
async def test_worker_announces_release_and_waits_on_incompatible_gateway(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from litellm.proxy.lens.release import PROTOCOL_VERSION

    monkeypatch.setenv("LITELLM_RELEASE_TAG", "v1.2.3")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/lens/worker/claim"
        assert request.url.params["protocol_version"] == str(PROTOCOL_VERSION)
        assert request.url.params["worker_release"] == "v1.2.3"
        return httpx.Response(409, json={"detail": "Upgrade the Lens worker to v1.2.4"})

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        assert not await LensWorker(client, analysis=analyze_sample).run_once()
    assert "Upgrade the Lens worker to v1.2.4" in caplog.text
