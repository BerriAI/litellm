import asyncio
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.engine.analysis import Candidate, Examined, evidence_valid, extract, investigate, partition_content
from litellm.proxy.engine.models import (
    Claim,
    Coverage,
    Evidence,
    Execution,
    ExecutionContent,
    ModelRequest,
    ModelResult,
    Sample,
    TracePart,
)
from litellm.proxy.engine.state import queue_job
from tests.unit.proxy.engine.test_state import NOW, engine, finding


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("complete", "cancel", "failure"))
async def test_parallel_review_shares_one_model_limit_and_cleans_up(outcome: str) -> None:
    from litellm.proxy.engine.analysis import ANALYSIS_CONCURRENCY, analyze_sample

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

    async def progress(stage: str, coverage: Coverage) -> None:
        if stage == "Reading executions":
            counts.put(coverage.screened)

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
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
            assert entered.qsize() == exited.qsize() == len(executions) * 2
            assert tuple(counts.get_nowait() for _ in range(counts.qsize())) == tuple(range(len(executions) + 1))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_chunk_extraction_is_concurrent_and_preserves_every_observation() -> None:
    from litellm.proxy.engine.analysis import Extraction, Observation

    execution: Final = Execution(
        id="run", source="traces", trace_id="trace", team_id="alpha", name="run", start_time="", span_count=6
    )
    parts: Final = tuple(
        TracePart(execution_id="run", span_id=str(i), name="tool", kind="tool", content=str(i) * 8000) for i in range(6)
    )
    arrived: Final = SimpleQueue[str]()
    both: Final = asyncio.Event()

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=parts)

    async def model(request: ModelRequest) -> ModelResult:
        part: Final = parts[0] if parts[0].content in request.prompt else parts[3]
        arrived.put(part.span_id)
        if arrived.qsize() == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=2)
        return ModelResult(
            content=Extraction(
                observations=(
                    Observation(
                        check_id="retries",
                        summary=part.span_id,
                        evidence=(Evidence(execution_id="run", span_id=part.span_id, quote=part.content[:10]),),
                    ),
                )
            ).model_dump_json(),
            cost=0,
        )

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert frozenset(o.summary for o in result.observations) == frozenset(("0", "3"))
    assert result.parts == parts


@pytest.mark.asyncio
async def test_independent_investigations_overlap_and_report_completions() -> None:
    from litellm.proxy.engine.analysis import investigate_candidates

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

    async def progress(stage: str, coverage: Coverage) -> None:
        assert stage == "Checking original evidence"
        progress_counts.put(coverage.investigated)

    candidates: Final = tuple(
        Candidate(check_id="retries", title=str(i), hypothesis="Investigate", execution_ids=()) for i in range(2)
    )
    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
    results: Final = tuple(
        [
            result
            async for result in investigate_candidates(
                claim, candidates, (), read, model, progress, Coverage(candidates=2)
            )
        ]
    )
    assert results == ()
    assert tuple(progress_counts.get_nowait() for _ in range(progress_counts.qsize())) == (1, 2)


def test_quote_must_match_the_claimed_execution_and_span() -> None:
    part: Final = TracePart(execution_id="run1", span_id="span", name="search", kind="tool", content="timeout")
    assert evidence_valid(Evidence(execution_id="run1", span_id="span", quote="timeout"), (part,))
    assert not evidence_valid(Evidence(execution_id="other", span_id="span", quote="timeout"), (part,))
    assert not evidence_valid(Evidence(execution_id="run1", span_id="other", quote="timeout"), (part,))
    assert not evidence_valid(Evidence(execution_id="run1", span_id="span", quote="success"), (part,))


def test_chunks_preserve_all_spans_and_keep_context_bounded() -> None:
    parts: Final = tuple(
        TracePart(execution_id="run", span_id=str(i), name="tool", kind="tool", content="x" * 8000) for i in range(10)
    )
    chunks: Final = partition_content(parts)
    assert tuple(len(chunk) for chunk in chunks) == (3, 3, 3, 1)
    assert sum(len(chunk) for chunk in chunks) == 10
    assert tuple(p.span_id for p in chunks[-1]) == ("9",)


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

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
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

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
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

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Retries", hypothesis="Unrecovered", execution_ids=("run1",)),
        (examined,),
        read,
        model,
    )
    assert result.finding == finding("run1")


@pytest.mark.asyncio
@pytest.mark.parametrize("quote", ["timeout", "invented quote"])
async def test_oversized_model_evidence_is_retried_and_quotes_still_verified(quote: str) -> None:
    execution: Final = Execution(
        id="run1", source="traces", trace_id="t", team_id="alpha", name="review", start_time="", span_count=1
    )
    part: Final = TracePart(execution_id="run1", span_id="span", name="tool", kind="tool", content="timeout")
    attempts: Final = iter((8, 1))

    async def read(_execution_id: str, _cursor: str, _offset: int) -> ExecutionContent:
        return ExecutionContent(execution=execution, parts=(part,))

    async def model(request: ModelRequest) -> ModelResult:
        count: Final = next(attempts)
        if count == 1:
            assert "validation errors" in request.prompt
            assert '"max_length":6' in request.prompt
        evidence: Final = Evidence(execution_id="run1", span_id="span", quote=quote).model_dump_json()
        return ModelResult(
            content='{"observations":[{"check_id":"retries","summary":"Tool timeout","evidence":['
            + ",".join(evidence for _ in range(count))
            + "]}]}",
            cost=0,
        )

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
    result: Final = await extract(claim, execution, read, model)
    assert len(result.observations) == (1 if quote == "timeout" else 0)
    assert next(attempts, None) is None


@pytest.mark.asyncio
async def test_invalid_model_output_has_only_one_repair_attempt() -> None:
    from pydantic import ValidationError

    from litellm.proxy.engine.analysis import Extraction, structured_response

    attempts: Final = iter((1, 2))

    async def model(_request: ModelRequest) -> ModelResult:
        assert next(attempts, None) is not None, "Model repair exceeded its retry limit"
        return ModelResult(content="not JSON", cost=0)

    with pytest.raises(ValidationError):
        await structured_response(ModelRequest(purpose="extract", prompt="Extract observations"), Extraction, model)
    assert next(attempts, None) is None


@pytest.mark.asyncio
async def test_grouping_consolidates_prior_batches_and_reports_real_progress() -> None:
    from litellm.proxy.engine.analysis import Clusters, Observation, cluster_batches
    from litellm.proxy.engine.models import Coverage

    candidate: Final = Candidate(
        check_id="retries", title="Outage", hypothesis="Tool unavailable", execution_ids=("run1",)
    )
    observation: Final = Observation(check_id="retries", summary="Repeated timeout", evidence=())
    stages: Final = iter((0, 1))
    calls: Final = iter((False, True))

    async def progress(stage: str, coverage: Coverage) -> None:
        assert stage == "Grouping observations"
        assert coverage.grouping_batches == 2
        assert coverage.grouped_batches == next(stages)
        assert coverage.screened == 2

    async def model(request: ModelRequest) -> ModelResult:
        if next(calls):
            assert '"previous_candidates": [{"check_id": "retries", "title": "Outage"' in request.prompt
            return ModelResult(
                content=Clusters(
                    candidates=(candidate.model_copy(update=MappingProxyType({"execution_ids": ("run1", "run2")})),)
                ).model_dump_json(),
                cost=0,
            )
        return ModelResult(content=Clusters(candidates=(candidate,)).model_dump_json(), cost=0)

    result: Final = await cluster_batches(
        ((observation,), (observation,)), model, progress, Coverage(screened=2, grouping_batches=2)
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
    decisions: Final = iter(("read", "submit"))

    async def model(request: ModelRequest) -> ModelResult:
        if next(decisions) == "read":
            return ModelResult(content='{"action":"read","execution_id":"run1","offset":8000}', cost=0)
        assert '"content": "timeout"' in request.prompt
        return ModelResult(content='{"action":"submit","finding":' + draft.model_dump_json() + "}", cost=0)

    async def read(execution_id: str, _cursor: str, offset: int) -> ExecutionContent:
        assert execution_id == "run1" and offset == 8000
        return ExecutionContent(execution=execution, parts=(later,))

    claim: Final = Claim(engine_id="engine", job=queue_job(engine(), NOW, "job").jobs[0], findings=())
    result: Final = await investigate(
        claim,
        Candidate(check_id="retries", title="Retries", hypothesis="Unrecovered", execution_ids=("run1",)),
        (examined,),
        read,
        model,
    )
    assert result.finding == draft
