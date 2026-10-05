import asyncio
import json
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.lens.analysis import Candidate, Examined, evidence_valid, extract, investigate, partition_content
from litellm.proxy.lens.models import (
    Activity,
    Claim,
    Coverage,
    Evidence,
    Execution,
    ExecutionContent,
    InFlight,
    ModelMessage,
    ModelRequest,
    ModelResult,
    Review,
    Sample,
    TracePart,
)
from litellm.proxy.lens.state import queue_job
from tests.unit.proxy.lens.test_state import NOW, finding, issue_brief, lens


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("complete", "cancel", "failure"))
async def test_parallel_review_shares_one_model_limit_and_cleans_up(outcome: str) -> None:
    from litellm.proxy.lens.analysis import ANALYSIS_CONCURRENCY, analyze_sample

    executions: Final = tuple(
        Execution(id=str(i), source="traces", trace_id=str(i), team_id="alpha", name="run", start_time="", span_count=6)
        for i in range(ANALYSIS_CONCURRENCY + 1)
    )
    entered: Final = SimpleQueue[str]()
    exited: Final = SimpleQueue[str]()
    reads: Final = SimpleQueue[str]()
    counts: Final = SimpleQueue[int]()
    saturated: Final = asyncio.Event()
    release: Final = asyncio.Event()
    stalled: Final = asyncio.Event()

    async def read(execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        reads.put(execution_id)
        execution: Final = next(e for e in executions if e.id == execution_id)
        return ExecutionContent(
            execution=execution,
            parts=tuple(
                TracePart(execution_id=execution_id, span_id=str(i), name="tool", kind="tool", content="x" * 8000)
                for i in range(6)
            ),
        )

    async def model(request: ModelRequest) -> ModelResult:
        entered.put(request.prompt)
        first: Final = entered.qsize() == 1
        assert entered.qsize() - exited.qsize() <= ANALYSIS_CONCURRENCY
        if entered.qsize() == ANALYSIS_CONCURRENCY:
            saturated.set()
        try:
            await release.wait()
            if outcome == "failure":
                if first:
                    raise ValueError("invalid model response")
                await stalled.wait()
            return ModelResult(content='{"observations":[]}', cost=0)
        finally:
            exited.put(request.prompt)

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        assert coverage is not None
        if stage == "Reading executions" and (_reading is None or _review is not None):
            counts.put(coverage.screened)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    task: Final = asyncio.create_task(
        analyze_sample(claim, Sample(executions=executions, eligible=len(executions)), read, model, progress)
    )
    try:
        await asyncio.wait_for(saturated.wait(), timeout=2)
        assert entered.qsize() == ANALYSIS_CONCURRENCY
        assert reads.qsize() == ANALYSIS_CONCURRENCY
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert entered.qsize() == exited.qsize() == ANALYSIS_CONCURRENCY
        elif outcome == "failure":
            release.set()
            with pytest.raises(ValueError, match="invalid model response"):
                await asyncio.wait_for(task, timeout=2)
            assert entered.qsize() == exited.qsize()
        else:
            release.set()
            result: Final = await task
            assert result.coverage.screened == len(executions)
            assert entered.qsize() == exited.qsize() == len(executions)
            assert tuple(counts.get_nowait() for _ in range(counts.qsize())) == tuple(range(len(executions) + 1))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_independent_investigations_overlap_and_report_completions() -> None:
    from litellm.proxy.lens.analysis import investigate_candidates

    arrived: Final = SimpleQueue[str]()
    progress_counts: Final = SimpleQueue[int]()
    both: Final = asyncio.Event()

    async def model(request: ModelRequest) -> ModelResult:
        arrived.put(request.prompt)
        if arrived.qsize() == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=2)
        return ModelResult(content='{"action":"inconclusive"}', cost=0)

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        pytest.fail("Inconclusive decisions must not fetch evidence")

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        assert coverage is not None
        assert stage == "Checking original evidence"
        progress_counts.put(coverage.investigated)

    candidates: Final = tuple(
        Candidate(check_id="retries", title=str(i), hypothesis="Investigate", execution_ids=()) for i in range(2)
    )
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    results: Final = tuple(
        [
            result
            async for result in investigate_candidates(
                claim, candidates, (), read, model, progress, Coverage(candidates=2)
            )
        ]
    )
    assert len(results) == 2
    assert all(result.finding is None for result in results)
    assert tuple(progress_counts.get_nowait() for _ in range(progress_counts.qsize())) == (1, 2)


def test_quote_must_match_the_claimed_execution_and_span() -> None:
    part: Final = TracePart(execution_id="run1", span_id="span", name="search", kind="tool", content="timeout")
    assert evidence_valid(Evidence(execution_id="run1", span_id="span", quote="timeout"), (part,))
    assert not evidence_valid(Evidence(execution_id="other", span_id="span", quote="timeout"), (part,))
    assert not evidence_valid(Evidence(execution_id="run1", span_id="other", quote="timeout"), (part,))
    assert not evidence_valid(Evidence(execution_id="run1", span_id="span", quote="success"), (part,))


def test_excerpt_omission_is_not_original_evidence() -> None:
    part: Final = TracePart(
        execution_id="run1",
        span_id="span",
        name="tool",
        kind="tool",
        content="Input: requested\n[... content omitted ...]\nOutput: failed",
        truncated=True,
    )
    assert evidence_valid(Evidence(execution_id="run1", span_id="span", quote="Output: failed"), (part,))
    assert not evidence_valid(Evidence(execution_id="run1", span_id="span", quote=part.content), (part,))
    assert not evidence_valid(Evidence(execution_id="run1", span_id="span", quote="[... content omitted ...]"), (part,))


@pytest.mark.asyncio
async def test_reviewer_sees_final_outcome_and_catalog_across_pages() -> None:
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="run", start_time="", span_count=2
    )
    root: Final = TracePart(execution_id="run", span_id="01", name="task", kind="agent", content="Task: write a report")
    editor: Final = TracePart(
        execution_id="run", span_id="02", parent_span_id="01", name="editor", kind="agent", content="Delivered report"
    )
    pages: Final = SimpleQueue[str]()

    async def read(_execution_id: str, cursor: str, _offset: int) -> ExecutionContent:
        pages.put(cursor)
        return ExecutionContent(
            execution=execution, parts=(editor,) if cursor else (root,), next_cursor=None if cursor else "01"
        )

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        assert payload["catalog_complete"] is True
        assert tuple(row[2] for row in payload["catalog"]) == ("task", "editor")
        assert "Delivered report" in request.prompt
        assert pages.qsize() == 2
        return ModelResult(content='{"observations":[],"cannot_assess":false}', cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert root in result.parts
    assert not result.cannot_assess


@pytest.mark.asyncio
async def test_reviewer_fetches_targeted_evidence_and_rejects_outside_catalog_reads() -> None:
    from litellm.proxy.lens.analysis import Observation, SpanRead, TraceReview

    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="run", start_time="", span_count=2
    )
    root: Final = TracePart(
        execution_id="run", span_id="01", name="task", kind="agent", content="Find the verified result"
    )
    preview: Final = TracePart(
        execution_id="run",
        span_id="02",
        parent_span_id="01",
        name="search",
        kind="tool",
        content="Long document prefix",
        truncated=True,
    )
    later: Final = preview.model_copy(
        update=MappingProxyType({"content": "Verified result: failed", "truncated": False})
    )
    calls: Final = iter((False, True))
    reads: Final = SimpleQueue[tuple[str, int]]()

    async def read(execution_id: str, cursor: str, offset: int) -> ExecutionContent:
        assert execution_id == "run"
        reads.put((cursor, offset))
        if offset:
            assert cursor == "01" and offset == 8000
            return ExecutionContent(execution=execution, parts=(later,))
        return ExecutionContent(execution=execution, parts=(root, preview), partial=True)

    async def model(request: ModelRequest) -> ModelResult:
        if not next(calls):
            return ModelResult(
                content=TraceReview(
                    reads=(SpanRead(span_id="02", offset=8000), SpanRead(span_id="foreign"))
                ).model_dump_json(),
                cost=0,
            )
        assert "Verified result: failed" in request.prompt
        return ModelResult(
            content=TraceReview(
                observations=(
                    Observation(
                        check_id="retries",
                        summary="Verified failure",
                        evidence=(Evidence(execution_id="run", span_id="02", quote="Verified result: failed"),),
                    ),
                )
            ).model_dump_json(),
            cost=0,
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert len(result.observations) == 1
    assert result.observations[0].evidence[0].quote == "Verified result: failed"
    assert tuple(reads.get_nowait() for _ in range(reads.qsize())) == (("", 0), ("01", 8000))


@pytest.mark.asyncio
async def test_reviewer_stops_repeated_read_requests() -> None:
    from litellm.proxy.lens.analysis import SpanRead, TraceReview

    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="run", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run", span_id="01", name="task", kind="agent", content="Partial export")
    reads: Final = SimpleQueue[int]()
    calls: Final = SimpleQueue[int]()

    async def read(_execution_id: str, _cursor: str, offset: int) -> ExecutionContent:
        reads.put(offset)
        return ExecutionContent(execution=execution, parts=(part,), partial=True)

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(1)
        if json.loads(request.prompt)["must_decide"]:
            return ModelResult(content='{"observations": [], "cannot_assess": true}', cost=0)
        return ModelResult(
            content=TraceReview(reads=(SpanRead(span_id="01"),), cannot_assess=True).model_dump_json(), cost=0
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert result.cannot_assess
    assert reads.qsize() == 2
    assert calls.qsize() == 3


def test_chunks_preserve_all_spans_and_keep_context_bounded() -> None:
    parts: Final = tuple(
        TracePart(execution_id="run", span_id=str(i), name="tool", kind="tool", content="x" * 8000) for i in range(10)
    )
    chunks: Final = partition_content(parts)
    assert all(len(json.dumps(tuple(p.model_dump() for p in chunk))) <= 24000 for chunk in chunks)
    assert tuple(p for chunk in chunks for p in chunk) == parts


@pytest.mark.asyncio
async def test_investigator_rejects_a_fabricated_quote() -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="search", start_time="", span_count=1
    )
    examined: Final = Examined(
        execution=execution,
        observations=(),
        parts=(TracePart(execution_id="run1", span_id="span", name="search", kind="tool", content="succeeded"),),
        partial=False,
        cannot_assess=False,
    )

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content='{"action":"submit","finding":' + finding("run1").model_dump_json() + "}", cost=0)

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=examined.parts)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Retries", hypothesis="Unrecovered", execution_ids=("run1",)),
        (examined,),
        read,
        model,
    )
    assert result.finding is None


@pytest.mark.asyncio
@pytest.mark.parametrize("paginated", [False, True])
@pytest.mark.parametrize("assessable", [False, True])
async def test_assessable_content_is_not_overridden_by_unknown_chunks(paginated: bool, assessable: bool) -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="review", start_time="", span_count=4
    )
    unknown: Final = tuple(
        TracePart(execution_id="run1", span_id=str(i), name="tool", kind="tool", content="x" * 8000) for i in range(3)
    )
    answer: Final = TracePart(
        execution_id="run1",
        span_id="3",
        name="agent",
        kind="agent",
        content="verified result" if assessable else "outcome unavailable",
    )

    async def read(_execution_id: str, cursor: str, _offset: int) -> ExecutionContent:
        if cursor:
            return ExecutionContent(execution=execution, parts=(answer,))
        return ExecutionContent(
            execution=execution,
            parts=unknown if paginated else (*unknown, answer),
            next_cursor="2" if paginated else None,
        )

    async def model(request: ModelRequest) -> ModelResult:
        unavailable: Final = "false" if "verified result" in request.prompt else "true"
        return ModelResult(content='{"observations":[],"cannot_assess":' + unavailable + "}", cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert result.cannot_assess is not assessable


@pytest.mark.asyncio
async def test_investigator_keeps_final_outcome_ahead_of_repeated_model_history() -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="review", start_time="", span_count=6
    )
    history: Final = tuple(
        TracePart(
            execution_id="run1", span_id=str(i), name="chat", kind="llm", parent_span_id="span", content="x" * 8000
        )
        for i in range(5)
    )
    outcome: Final = TracePart(execution_id="run1", span_id="span", name="lead", kind="agent", content="timeout")
    examined: Final = Examined(
        execution=execution, observations=(), parts=(*history, outcome), partial=False, cannot_assess=False
    )

    async def model(request: ModelRequest) -> ModelResult:
        if '"content": "timeout"' not in request.prompt:
            return ModelResult(content='{"action":"inconclusive"}', cost=0)
        return ModelResult(content='{"action":"submit","finding":' + finding("run1").model_dump_json() + "}", cost=0)

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=examined.parts)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Retries", hypothesis="Unrecovered", execution_ids=("run1",)),
        (examined,),
        read,
        model,
    )
    assert result.finding == finding("run1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quote, check_id, accepted",
    [("timeout", "retries", True), ("invented quote", "retries", False), ("timeout", "unknown", False)],
)
async def test_many_model_citations_are_accepted_but_quotes_are_still_verified(
    quote: str, check_id: str, accepted: bool
) -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="review", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run1", span_id="span", name="tool", kind="tool", content="timeout")
    attempts: Final = iter((8,))

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=(part,))

    async def model(request: ModelRequest) -> ModelResult:
        count: Final = next(attempts)
        evidence: Final = Evidence(execution_id="run1", span_id="span", quote=quote).model_dump_json()
        return ModelResult(
            content='{"observations":[{"check_id":"'
            + check_id
            + '","summary":"Tool timeout","evidence":['
            + ",".join(evidence for _ in range(count))
            + "]}]}",
            cost=0,
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert len(result.observations) == int(accepted)
    assert result.cannot_assess is not accepted
    assert next(attempts, None) is None


@pytest.mark.asyncio
async def test_invalid_model_output_has_only_one_repair_attempt() -> None:
    from litellm.proxy.lens.analysis import AnalysisResponseError, Extraction, structured_response

    attempts: Final = iter((1, 2))

    async def model(_request: ModelRequest) -> ModelResult:
        assert next(attempts, None) is not None, "Model repair exceeded its retry limit"
        return ModelResult(content="not JSON", cost=0)

    with pytest.raises(
        AnalysisResponseError, match="Reading executions failed: Extraction response invalid after 2 attempts"
    ):
        await structured_response(ModelRequest(purpose="extract", prompt="Extract observations"), Extraction, model)
    assert next(attempts, None) is None


@pytest.mark.asyncio
async def test_async_validation_source_failure_propagates_without_a_model_repair() -> None:
    from litellm.proxy.lens.analysis import Extraction, structured_response

    calls: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(request)
        return ModelResult(content=Extraction().model_dump_json(), cost=0)

    async def validate(_result: Extraction) -> str | None:
        raise ValueError("Evidence source is unavailable")

    with pytest.raises(ValueError, match="Evidence source is unavailable"):
        await structured_response(
            ModelRequest(purpose="extract", prompt="Extract observations"), Extraction, model, validate
        )
    assert calls.qsize() == 1


@pytest.mark.asyncio
async def test_conversation_repair_appends_raw_response_and_correction_without_changing_the_prefix() -> None:
    from litellm.proxy.lens.analysis import Extraction, structured_response_with_history

    original: Final = ModelRequest(
        purpose="extract",
        prompt="Stable task",
        messages=(ModelMessage(role="system", content="Stable task"), ModelMessage(role="user", content="Evidence")),
    )
    malformed: Final = '{ "observations": "wrong type" }'
    corrected: Final = '{ "observations": [], "cannot_assess": false }'
    attempts: Final = iter((0, 1))
    repairs: Final = SimpleQueue[ModelRequest]()

    async def model(request: ModelRequest) -> ModelResult:
        if next(attempts) == 0:
            assert request == original
            return ModelResult(content=malformed, cost=0)
        assert request.prompt == original.prompt
        assert request.messages[:-2] == original.messages
        assert request.messages[-2] == ModelMessage(role="assistant", content=malformed)
        assert request.messages[-1].role == "system"
        assert "observations" in request.messages[-1].content
        repairs.put(request)
        return ModelResult(content=corrected, cost=0)

    result, history = await structured_response_with_history(original, Extraction, model)
    assert result == Extraction()
    assert history == (*repairs.get_nowait().messages, ModelMessage(role="assistant", content=corrected))
    assert next(attempts, None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", (False, True))
async def test_repair_repeats_complete_schema_without_unknown_fields_or_input_values(conversation: bool) -> None:
    from litellm.proxy.lens.analysis import Extraction, structured_response

    original: Final = ModelRequest(
        purpose="extract",
        prompt="Review original evidence",
        messages=(ModelMessage(role="system", content="Review original evidence"),) if conversation else (),
    )
    attempts: Final = iter((0, 1))

    async def model(request: ModelRequest) -> ModelResult:
        if next(attempts) == 0:
            return ModelResult(content='{"private_field_sentinel":"private_value_sentinel"}', cost=0)
        assert request.messages[-1].role == "system"
        assert request.messages[:-2] == original.conversation()
        content: Final = request.messages[-1].content
        correction: Final = TypeAdapter(dict[str, JsonValue]).validate_json(content)
        assert correction["response_schema"] == Extraction.model_json_schema()
        assert "extra_forbidden" in content
        assert "private_field_sentinel" not in content
        assert "private_value_sentinel" not in content
        return ModelResult(content=Extraction().model_dump_json(), cost=0)

    assert await structured_response(original, Extraction, model) == Extraction()
    assert next(attempts, None) is None


@pytest.mark.asyncio
async def test_grouping_consolidates_prior_batches_and_reports_real_progress() -> None:
    from litellm.proxy.lens.analysis import Clusters, Observation, cluster_batches
    from litellm.proxy.lens.models import Coverage

    candidate: Final = Candidate(
        check_id="retries", title="Outage", hypothesis="Tool unavailable", execution_ids=("run1",)
    )
    observations: Final = tuple(
        Observation(
            check_id="retries",
            summary="Repeated timeout",
            evidence=(Evidence(execution_id=identity, span_id="s", quote="timeout"),),
        )
        for identity in ("run1", "run2")
    )
    stages: Final = iter((0, 1))

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        assert coverage is not None
        assert stage == "Grouping observations"
        assert coverage.grouping_batches == 2
        assert coverage.grouped_batches == next(stages)
        assert coverage.screened == 2

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        references: Final = tuple(c["execution_ids"][0] for c in payload["candidates"])
        return ModelResult(
            content=Clusters(
                candidates=(candidate.model_copy(update=MappingProxyType({"execution_ids": references})),)
            ).model_dump_json(),
            cost=0,
        )

    result: Final = await cluster_batches(
        tuple((o,) for o in observations), model, progress, Coverage(screened=2, grouping_batches=2)
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].execution_ids == ("run1", "run2")
    assert next(stages, None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("later_span", ("later", "0"))
async def test_investigator_can_cite_a_later_page_or_offset(later_span: str) -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="review", start_time="", span_count=7
    )
    initial: Final = tuple(
        TracePart(execution_id="run1", span_id=str(i), name="agent", kind="agent", content="x" * 8000) for i in range(6)
    )
    later: Final = TracePart(execution_id="run1", span_id=later_span, name="tool", kind="tool", content="timeout")
    examined: Final = Examined(execution=execution, observations=(), parts=initial, partial=True, cannot_assess=False)
    draft: Final = finding("run1").model_copy(
        update={"evidence": (Evidence(execution_id="run1", span_id=later_span, quote="timeout"),)}
    )
    offsets: Final = iter((8000, 16000, None))

    async def model(request: ModelRequest) -> ModelResult:
        offset: Final = next(offsets)
        if offset is not None:
            return ModelResult(content=json.dumps({"action": "read", "execution_id": "run1", "offset": offset}), cost=0)
        assert json.loads(request.prompt)["must_decide"] is False
        assert '"content": "timeout"' in request.prompt
        return ModelResult(content='{"action":"submit","finding":' + draft.model_dump_json() + "}", cost=0)

    async def read(execution_id: str, _cursor: str, offset: int) -> ExecutionContent:
        assert execution_id == "run1" and offset in (8000, 16000)
        return ExecutionContent(execution=execution, parts=(later,))

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Retries", hypothesis="Unrecovered", execution_ids=("run1",)),
        (examined,),
        read,
        model,
    )
    assert result.finding == draft


@pytest.mark.asyncio
async def test_thousands_of_matching_runs_keep_all_members_without_a_growing_model_prompt() -> None:
    from litellm.proxy.lens.analysis import Clusters, Observation, cluster_batches, observation_batches

    observations: Final = tuple(
        Observation(
            check_id="retries",
            summary="Lookup failed without recovery",
            evidence=(Evidence(execution_id=f"execution-{index}", span_id="lookup", quote="timeout"),),
        )
        for index in range(2501)
    )
    counts: Final = SimpleQueue[int]()

    async def model(request: ModelRequest) -> ModelResult:
        assert len(request.prompt) < 40000
        payload: Final = json.loads(request.prompt)
        return ModelResult(
            content=Clusters(
                candidates=(
                    Candidate(
                        check_id="retries",
                        title="Lookup unavailable",
                        hypothesis="Unrecovered timeout",
                        execution_ids=tuple(c["execution_ids"][0] for c in payload["candidates"]),
                    ),
                )
            ).model_dump_json(),
            cost=0,
        )

    async def progress(
        _stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        assert coverage is not None
        counts.put(coverage.grouped_batches)

    batches: Final = observation_batches(observations)
    result: Final = await cluster_batches(batches, model, progress, Coverage(grouping_batches=len(batches)))
    assert len(result.candidates) == 1
    assert frozenset(result.candidates[0].execution_ids) == frozenset(f"execution-{i}" for i in range(2501))
    assert counts.qsize() == len(batches)


@pytest.mark.asyncio
async def test_grouping_preserves_observations_omitted_by_model() -> None:
    from litellm.proxy.lens.analysis import merge_candidates

    original: Final = Candidate(
        check_id="retries", title="Unrecovered failure", hypothesis="Timeout", execution_ids=("run",)
    )

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content='{"candidates":[]}', cost=0)

    incoming, retained = await merge_candidates((original,), 0, model)
    assert incoming == (original,)
    assert retained == ()


@pytest.mark.asyncio
async def test_grouping_repairs_duplicate_members_before_creating_findings() -> None:
    from litellm.proxy.lens.analysis import Clusters, merge_candidates

    original: Final = Candidate(
        check_id="retries", title="Unrecovered failure", hypothesis="Timeout", execution_ids=("run",)
    )
    attempts: Final = iter((2, 1))

    async def model(request: ModelRequest) -> ModelResult:
        copies: Final = next(attempts)
        if copies == 1:
            assert "do not duplicate" in request.messages[-1].content
            assert request.messages[-1].role == "system"
        group: Final = original.model_copy(update=MappingProxyType({"execution_ids": ("p0",)}))
        return ModelResult(content=Clusters(candidates=(group,) * copies).model_dump_json(), cost=0)

    incoming, retained = await merge_candidates((original,), 0, model)
    assert incoming == (original,)
    assert retained == ()
    assert next(attempts, None) is None


@pytest.mark.asyncio
async def test_review_keeps_original_ids_in_per_run_assessments() -> None:
    from litellm.proxy.lens.analysis import analyze_sample

    execution: Final = Execution(
        id="opaque-original-id",
        source="requests",
        trace_id="request",
        team_id="",
        name="call",
        start_time="",
        span_count=1,
    )

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        assert identity == execution.id
        return ExecutionContent(
            execution=execution,
            parts=(
                TracePart(execution_id=identity, span_id="root", name="call", kind="llm", content="Task completed"),
            ),
        )

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content='{"observations":[],"cannot_assess":false}', cost=0)

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        pass

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await analyze_sample(claim, Sample(executions=(execution,), eligible=1), read, model, progress)
    assert result.assessments[0].execution_id == execution.id
    assert not result.assessments[0].cannot_assess
    assert result.coverage.screened == 1


@pytest.mark.asyncio
async def test_investigation_context_accounts_for_metadata_on_thousands_of_short_spans() -> None:
    executions: Final = tuple(
        Execution(
            id=f"run-{i}",
            source="traces",
            trace_id=f"trace-{i}",
            team_id="",
            name="Short successful task",
            start_time="",
            span_count=1,
        )
        for i in range(2501)
    )
    examined: Final = tuple(
        Examined(
            execution=e,
            observations=(),
            parts=(TracePart(execution_id=e.id, span_id="root", name="task", kind="agent", content="Done"),),
            partial=False,
            cannot_assess=False,
        )
        for e in executions
    )

    async def model(request: ModelRequest) -> ModelResult:
        assert len(request.prompt) < 100000
        payload: Final = json.loads(request.prompt)
        assert payload["candidate_run_count"] == 2501
        assert payload["catalog_pages"] > 1
        return ModelResult(content='{"action":"inconclusive"}', cost=0)

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        pytest.fail("No read was requested")

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(
            check_id="retries",
            title="Success",
            hypothesis="Successful recovery",
            execution_ids=tuple(e.id for e in executions),
        ),
        examined,
        read,
        model,
    )
    assert result.finding is None


@pytest.mark.asyncio
async def test_completed_read_does_not_make_supported_review_unknown() -> None:
    from litellm.proxy.lens.analysis import Observation, SpanRead, TraceReview

    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run", span_id="s", name="task", kind="agent", content="timeout")
    observation: Final = Observation(
        check_id="retries", summary="Failed", evidence=(Evidence(execution_id="run", span_id="s", quote="timeout"),)
    )
    calls: Final = SimpleQueue[int]()

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=(part,))

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(1)
        if json.loads(request.prompt)["must_decide"]:
            return ModelResult(
                content=json.dumps({"observations": [observation.model_dump()], "cannot_assess": False}), cost=0
            )
        return ModelResult(
            content=TraceReview(reads=(SpanRead(span_id="s"),), observations=(observation,)).model_dump_json(), cost=0
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert result.observations == (observation,)
    assert not result.cannot_assess and not result.partial
    assert calls.qsize() == 3


@pytest.mark.asyncio
async def test_echoed_feedback_page_does_not_skip_requested_evidence() -> None:
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    requests: Final = SimpleQueue[int]()

    async def read(_identity: str, _cursor: str, offset: int) -> ExecutionContent:
        requests.put(offset)
        return ExecutionContent(
            execution=execution,
            parts=(
                TracePart(
                    execution_id="run",
                    span_id="s",
                    name="task",
                    kind="agent",
                    content="timeout" if offset else "abbreviated",
                    truncated=not offset,
                ),
            ),
        )

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        if not payload["read_evidence"]:
            return ModelResult(content='{"feedback_page":0,"reads":[{"span_id":"s","offset":1}]}', cost=0)
        return ModelResult(
            content=json.dumps(
                {
                    "feedback_page": 0,
                    "observations": [
                        {
                            "check_id": "retries",
                            "summary": "Timed out",
                            "evidence": [{"execution_id": "run", "span_id": "s", "quote": "timeout"}],
                        }
                    ],
                }
            ),
            cost=0,
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert tuple(requests.get_nowait() for _ in range(requests.qsize())) == (0, 1)
    assert len(result.observations) == 1
    assert result.observations[0].evidence[0].quote == "timeout"
    assert not result.partial and not result.cannot_assess


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("catalog", "observations", "feedback", "read"))
async def test_empty_navigation_requires_a_final_decision(action: str) -> None:
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    examined: Final = Examined(execution=execution, observations=(), parts=(), partial=False, cannot_assess=False)
    calls: Final = SimpleQueue[int]()

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=())

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(1)
        assert calls.qsize() <= 2
        if json.loads(request.prompt)["must_decide"]:
            return ModelResult(content='{"action":"inconclusive"}', cost=0)
        return ModelResult(content=json.dumps({"action": action, "page": 999, "execution_id": "run"}), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Timeout", hypothesis="Failed", execution_ids=("run",)),
        (examined,),
        read,
        model,
    )
    assert result.finding is None
    assert calls.qsize() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("extract", "investigate"))
async def test_large_feedback_history_is_accessible_without_overflowing_context(phase: str) -> None:
    from litellm.proxy.lens.state import merge_finding

    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run", span_id="span", name="task", kind="agent", content="timeout")
    accepted: Final = merge_finding(lens(), finding("run"), 1, NOW)
    prior: Final = tuple(
        accepted.model_copy(
            update=MappingProxyType({"id": str(i), "status": "dismissed", "reason": f"Accepted-{i}: " + "x" * 1900})
        )
        for i in range(60)
    )
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=prior)
    pages: Final = SimpleQueue[int]()

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=(part,))

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        assert len(request.prompt) < 50000
        pages.put(payload["feedback_page"])
        last: Final = payload["feedback_pages"] - 1
        if payload["feedback_page"] == 0:
            return ModelResult(
                content=json.dumps(
                    {"feedback_page": last} if phase == "extract" else {"action": "feedback", "page": last}
                ),
                cost=0,
            )
        assert "Accepted-59" in request.prompt
        return ModelResult(content='{"observations":[]}' if phase == "extract" else '{"action":"inconclusive"}', cost=0)

    if phase == "extract":
        result: Final = await extract(claim, execution, read, model)
        assert not result.observations
    else:
        investigated: Final = await investigate(
            claim,
            Candidate(check_id="retries", title="Timeout", hypothesis="Failed", execution_ids=("run",)),
            (Examined(execution=execution, observations=(), parts=(part,), partial=False, cannot_assess=False),),
            read,
            model,
        )
        assert investigated.finding is None
    assert pages.qsize() == 2
    assert pages.get_nowait() == 0
    assert pages.get_nowait() > 0


@pytest.mark.asyncio
async def test_final_registry_reconciles_patterns_split_across_pages() -> None:
    from litellm.proxy.lens.analysis import Clusters, Observation, cluster_batches

    observations: Final = tuple(
        Observation(
            check_id="retries",
            summary=("timeout " + "x" * 1800),
            evidence=(Evidence(execution_id=f"run{i}", span_id="s", quote="timeout"),),
        )
        for i in range(20)
    )
    calls: Final = SimpleQueue[int]()

    async def model(request: ModelRequest) -> ModelResult:
        calls.put(1)
        payload: Final = json.loads(request.prompt)
        candidates: Final = tuple(Candidate.model_validate(c) for c in payload["candidates"])
        grouped: Final = (
            candidates
            if calls.qsize() == 1
            else (
                candidates[0].model_copy(
                    update=MappingProxyType({"execution_ids": tuple(c.execution_ids[0] for c in candidates)})
                ),
            )
        )
        return ModelResult(content=Clusters(candidates=grouped).model_dump_json(), cost=0)

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        return None

    result: Final = await cluster_batches((observations,), model, progress, Coverage())
    assert len(result.candidates) == 1
    assert frozenset(result.candidates[0].execution_ids) == frozenset(f"run{i}" for i in range(20))


@pytest.mark.asyncio
async def test_distinct_patterns_are_consolidated_in_batches_without_losing_runs() -> None:
    from litellm.proxy.lens.analysis import Observation, cluster_batches, observation_batches

    observations: Final = tuple(
        Observation(
            check_id="retries",
            summary=f"Distinct problem {i}: " + "details " * 40,
            evidence=(Evidence(execution_id=f"run{i}", span_id="s", quote="timeout"),),
        )
        for i in range(100)
    )
    requests: Final = SimpleQueue[int]()

    async def model(request: ModelRequest) -> ModelResult:
        requests.put(1)
        payload: Final = json.loads(request.prompt)
        return ModelResult(content=json.dumps({"candidates": payload["candidates"]}), cost=0)

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        pass

    result: Final = await cluster_batches(observation_batches(observations), model, progress, Coverage())
    assert len(result.candidates) == 100
    assert frozenset(c.execution_ids[0] for c in result.candidates) == frozenset(f"run{i}" for i in range(100))
    assert requests.qsize() < len(observations)


@pytest.mark.asyncio
async def test_invalid_candidate_response_preserves_other_findings_and_reports_inconclusive() -> None:
    from litellm.proxy.lens.analysis import investigate_candidates

    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run", span_id="span", name="tool", kind="tool", content="timeout")
    item: Final = Examined(execution=execution, observations=(), parts=(part,), partial=False, cannot_assess=False)
    candidates: Final = tuple(
        Candidate(check_id="retries", title=title, hypothesis="Failure", execution_ids=("run",))
        for title in ("Valid", "Malformed")
    )
    counts: Final = SimpleQueue[int]()

    async def read(_identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=())

    async def model(request: ModelRequest) -> ModelResult:
        if '"title": "Malformed"' in request.prompt:
            return ModelResult(content="not JSON", cost=0)
        return ModelResult(content=json.dumps({"action": "submit", "finding": finding("run").model_dump()}), cost=0)

    async def progress(
        _stage: str | None,
        coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        assert coverage is not None
        counts.put(coverage.inconclusive)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    results: Final = tuple(
        [
            result
            async for result in investigate_candidates(claim, candidates, (item,), read, model, progress, Coverage())
        ]
    )
    assert tuple(result.finding for result in results if result.finding is not None) == (finding("run"),)
    assert sum(result.finding is None for result in results) == 1
    assert "[json_invalid]" in next(result.error for result in results if result.finding is None)
    assert max(counts.get_nowait() for _ in range(counts.qsize())) == 1


@pytest.mark.asyncio
async def test_investigator_keeps_the_issue_brief() -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="search", start_time="", span_count=1
    )
    examined: Final = Examined(
        execution=execution,
        observations=(),
        parts=(TracePart(execution_id="run1", span_id="span", name="search", kind="tool", content="timeout"),),
        partial=False,
        cannot_assess=False,
    )
    draft: Final = finding("run1").model_copy(update={"brief": issue_brief("No repo tool")})

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult(content='{"action":"submit","finding":' + draft.model_dump_json() + "}", cost=0)

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=examined.parts)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Retries", hypothesis="Unrecovered", execution_ids=("run1",)),
        (examined,),
        read,
        model,
    )
    assert result.finding is not None
    assert result.finding.brief == draft.brief


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", (None, "length", "content_filter"))
async def test_grouping_failure_keeps_validation_details_without_model_content(finish_reason: str | None) -> None:
    from litellm.proxy.lens.analysis import AnalysisResponseError, Clusters, structured_response

    async def model(_request: ModelRequest) -> ModelResult:
        return ModelResult.model_validate(
            {"content": '{"candidates":[{"title":"private trace"}]}', "cost": 0, "finish_reason": finish_reason}
        )

    with pytest.raises(AnalysisResponseError) as caught:
        await structured_response(ModelRequest(purpose="cluster", prompt="private evidence"), Clusters, model)
    message: Final = str(caught.value)
    assert message.startswith("Grouping observations failed: Clusters response invalid after 2 attempts.")
    assert "candidates.0.check_id: Field required [missing]" in message
    assert "private" not in message
    if finish_reason:
        assert f"finish_reason={finish_reason}" in message
    else:
        assert "truncated" not in message


@pytest.mark.asyncio
async def test_truncated_but_valid_json_is_repaired_before_accepting_findings() -> None:
    from litellm.proxy.lens.analysis import Clusters, structured_response

    outputs: Final = iter(
        (
            ModelResult(content='{"candidates":[]}', cost=0, finish_reason="length"),
            ModelResult(content='{"candidates":[]}', cost=0),
        )
    )

    async def model(_request: ModelRequest) -> ModelResult:
        return next(outputs)

    assert await structured_response(ModelRequest(purpose="cluster", prompt="group"), Clusters, model) == Clusters()
    assert next(outputs, None) is None


@pytest.mark.asyncio
async def test_large_context_and_long_verified_quotes_do_not_silently_end_investigation() -> None:
    from litellm.proxy.lens.models import FindingDraft, LensSettings

    context: Final = "Read all recorded evidence. " * 5000
    long_quote: Final = "timeout detail " * 200
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run", span_id="span", name="tool", kind="tool", content=long_quote)
    reviewed: Final = Examined(execution=execution, observations=(), parts=(part,), partial=False, cannot_assess=False)
    expected: Final = FindingDraft.model_validate(
        {
            **finding("run").model_dump(),
            "description": "Recorded failure detail. " * 300,
            "evidence": [{"execution_id": "run", "span_id": "span", "quote": long_quote}],
        }
    )
    settings: Final = LensSettings.model_validate({**lens().settings.model_dump(), "context": context})
    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job", settings=settings).jobs[0], findings=())

    async def model(request: ModelRequest) -> ModelResult:
        assert json.loads(request.prompt)["context"] == context
        return ModelResult(content=json.dumps({"action": "submit", "finding": expected.model_dump()}), cost=0)

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        pytest.fail("Already supplied evidence should not require a read")

    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Failure", hypothesis="Retry failed", execution_ids=("run",)),
        (reviewed,),
        read,
        model,
    )
    assert result.finding == expected


@pytest.mark.asyncio
async def test_reviewer_can_read_every_offset_of_a_long_span_before_deciding() -> None:
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=1
    )
    original: Final = "trace evidence! " * 16000 + "late verified failure"
    offsets: Final = SimpleQueue[int]()
    seen: Final = SimpleQueue[str]()

    async def read(_execution_id: str, _cursor: str, offset: int) -> ExecutionContent:
        offsets.put(offset)
        content: Final = (
            "Preview; read for complete content" if offset == 0 else original[offset - 1 : offset - 1 + 8000]
        )
        return ExecutionContent(
            execution=execution,
            parts=(
                TracePart(
                    execution_id="run",
                    span_id="span",
                    name="agent",
                    kind="agent",
                    content=content,
                    truncated=offset == 0 or offset - 1 + 8000 < len(original),
                ),
            ),
        )

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        read_count: Final = payload["completed_read_count"]
        if read_count:
            seen.put(payload["read_evidence"][0]["content"])
        if read_count * 8000 < len(original):
            return ModelResult(
                content=json.dumps({"reads": [{"span_id": "span", "offset": 1 + read_count * 8000}]}), cost=0
            )
        return ModelResult(
            content=json.dumps(
                {
                    "observations": [
                        {
                            "check_id": "retries",
                            "summary": "Late failure",
                            "evidence": [{"execution_id": "run", "span_id": "span", "quote": "late verified failure"}],
                        }
                    ]
                }
            ),
            cost=0,
        )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert "".join(seen.get_nowait() for _ in range(seen.qsize())) == original
    assert tuple(offsets.get_nowait() for _ in range(offsets.qsize())) == (0, *range(1, len(original) + 1, 8000))
    assert result.observations[0].evidence[0].quote == "late verified failure"
    assert not result.cannot_assess


@pytest.mark.asyncio
async def test_investigator_can_read_all_evidence_pages_across_successive_span_batches() -> None:
    execution: Final = Execution(
        id="run", source="traces", trace_id="t", team_id="", name="task", start_time="", span_count=80
    )
    parts: Final = tuple(
        TracePart(
            execution_id="run",
            span_id=f"span{i:03}",
            parent_span_id="root",
            name=f"Step {i}",
            kind="tool",
            content="recorded evidence " * 400 + ("timeout" if i == 79 else "complete"),
        )
        for i in range(80)
    )
    seen: Final = SimpleQueue[str]()
    read_cursors: Final = SimpleQueue[str]()
    expected: Final = finding("run").model_copy(
        update={"evidence": (Evidence(execution_id="run", span_id="span079", quote="timeout"),)}
    )

    async def read(_identity: str, cursor: str, _offset: int) -> ExecutionContent:
        read_cursors.put(cursor)
        assert cursor in ("", "span039")
        return ExecutionContent(
            execution=execution,
            parts=parts[:40] if not cursor else parts[40:],
            next_cursor="span039" if not cursor else None,
        )

    async def model(request: ModelRequest) -> ModelResult:
        payload: Final = json.loads(request.prompt)
        if not payload["completed_read_count"]:
            return ModelResult(content=json.dumps({"action": "read", "execution_id": "run"}), cost=0)
        for part in payload["evidence"]:
            seen.put(part["span_id"])
        if payload["evidence_page"] + 1 < payload["evidence_pages"]:
            return ModelResult(content=json.dumps({"action": "evidence", "page": payload["evidence_page"] + 1}), cost=0)
        if payload["last_read"]["next_cursor"]:
            return ModelResult(
                content=json.dumps(
                    {"action": "read", "execution_id": "run", "cursor": payload["last_read"]["next_cursor"]}
                ),
                cost=0,
            )
        return ModelResult(content=json.dumps({"action": "submit", "finding": expected.model_dump()}), cost=0)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Failure", hypothesis="Failure", execution_ids=("run",)),
        (Examined(execution=execution, observations=(), parts=(), partial=False, cannot_assess=False),),
        read,
        model,
    )
    assert result.finding == expected
    assert result.error == ""
    assert tuple(seen.get_nowait() for _ in range(seen.qsize())) == tuple(p.span_id for p in parts)
    assert tuple(read_cursors.get_nowait() for _ in range(read_cursors.qsize())) == ("", "span039")


def test_review_flags_only_spans_cited_by_this_runs_observations() -> None:
    from litellm.proxy.lens.analysis import Observation, review_of

    execution: Final = Execution(
        id="run1",
        source="traces",
        trace_id="trace-1",
        team_id="",
        name="task",
        start_time="",
        span_count=3,
        service="bot",
    )
    shown: Final = tuple(
        TracePart(execution_id="run1", span_id=span, name=span, kind="tool", content=f"{span} output")
        for span in ("root", "search", "answer")
    )
    observation: Final = Observation(
        check_id="retries",
        summary="Search failed twice",
        evidence=(
            Evidence(execution_id="run1", span_id="search", quote="search output"),
            Evidence(execution_id="other", span_id="answer", quote="answer output"),
        ),
    )
    examined: Final = Examined(
        execution=execution,
        observations=(observation,),
        parts=shown,
        partial=False,
        cannot_assess=False,
        reasoning="Asked to search; it retried without recovering.",
        shown=shown,
    )
    review: Final = review_of(examined, "cerebras/model", 42, NOW)
    assert tuple((s.span_id, s.cited) for s in review.spans) == (("root", False), ("search", True), ("answer", False))
    assert (review.agent, review.trace_id, review.duration_ms) == ("bot", "trace-1", 42)
    assert review.reasoning == examined.reasoning
    assert tuple((v.check_id, v.summary) for v in review.verdicts) == (("retries", "Search failed twice"),)


@pytest.mark.asyncio
async def test_each_screened_run_reports_a_review_with_the_models_reasoning() -> None:
    from litellm.proxy.lens.analysis import analyze_sample

    execution: Final = Execution(
        id="opaque-original", source="traces", trace_id="trace", team_id="", name="task", start_time="", span_count=2
    )
    reasoning: Final = "The user asked for a refund; the tool timed out and the agent gave up."
    reviews: Final = SimpleQueue[Review]()

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(
            execution=execution,
            parts=(
                TracePart(execution_id=identity, span_id="a-root", name="agent", kind="agent", content="Refund please"),
                TracePart(
                    execution_id=identity,
                    span_id="b-tool",
                    parent_span_id="a-root",
                    name="refund",
                    kind="tool",
                    content="Tool timeout",
                ),
            ),
        )

    async def model(request: ModelRequest) -> ModelResult:
        if request.purpose == "cluster":
            return ModelResult(content='{"candidates":[]}', cost=0)
        if request.purpose == "investigate":
            return ModelResult(content='{"action":"inconclusive"}', cost=0)
        return ModelResult(
            content=json.dumps(
                {
                    "reasoning": reasoning,
                    "observations": [
                        {
                            "check_id": "retries",
                            "summary": "Gave up after a timeout",
                            "evidence": [{"execution_id": "r0", "span_id": "b-tool", "quote": "Tool timeout"}],
                        }
                    ],
                }
            ),
            cost=0,
        )

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if review is not None:
            reviews.put(review)

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    await analyze_sample(claim, Sample(executions=(execution,), eligible=1), read, model, progress)
    review: Final = reviews.get_nowait()
    assert reviews.empty()
    assert review.execution_id == execution.id
    assert review.reasoning == reasoning
    assert review.model == claim.job.settings.model
    assert tuple((s.span_id, s.cited) for s in review.spans) == (("a-root", False), ("b-tool", True))
    assert tuple(v.summary for v in review.verdicts) == ("Gave up after a timeout",)


@pytest.mark.asyncio
async def test_a_run_is_reported_in_flight_under_its_original_id_until_its_review_arrives() -> None:
    from litellm.proxy.lens.analysis import analyze_sample

    execution: Final = Execution(
        id="opaque-original", source="traces", trace_id="trace", team_id="", name="task", start_time="", span_count=1
    )
    reports: Final = SimpleQueue[tuple[str | None, tuple[str, ...] | None]]()

    async def read(identity: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(
            execution=execution,
            parts=(TracePart(execution_id=identity, span_id="s", name="agent", kind="agent", content="Hi"),),
        )

    async def model(request: ModelRequest) -> ModelResult:
        if request.purpose == "cluster":
            return ModelResult(content='{"candidates":[]}', cost=0)
        return ModelResult(content='{"observations":[]}', cost=0)

    async def progress(
        stage: str | None,
        _coverage: Coverage | None,
        review: Review | None = None,
        reading: tuple[InFlight, ...] | None = None,
        _activity: Activity | None = None,
        /,
    ) -> None:
        if stage == "Reading executions":
            reports.put(
                (
                    review and review.execution_id,
                    None if reading is None else tuple(f"{r.execution_id}:{r.trace_id}" for r in reading),
                )
            )

    claim: Final = Claim(lens_id="lens", job=queue_job(lens(), NOW, "job").jobs[0], findings=())
    await analyze_sample(claim, Sample(executions=(execution,), eligible=1), read, model, progress)
    assert tuple(reports.get_nowait() for _ in range(reports.qsize())) == (
        (None, None),
        (None, ("opaque-original:trace",)),
        ("opaque-original", ()),
    )
