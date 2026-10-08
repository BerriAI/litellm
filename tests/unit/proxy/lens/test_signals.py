import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from itertools import chain
from types import MappingProxyType, SimpleNamespace
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy.lens.models import Execution, Scope, TraceIdentity
from litellm.proxy.lens.repository import Database, Row
from litellm.proxy.lens.signal_repository import SignalRepository
from litellm.proxy.lens.signals import (
    DEFAULT_SIGNALS,
    SIGNAL_BACKLOG_SWEEP,
    SIGNAL_CLAIM_LEASE,
    SIGNAL_LIVE_SWEEP,
    SIGNAL_MAX_PER_TICK,
    SIGNAL_MAX_SCAN_PAGES,
    SIGNAL_TASK,
    DecisionQuestions,
    DecisionState,
    Signal,
    SignalAttempt,
    SignalClassifier,
    SignalConfig,
    SignalData,
    SignalStep,
    SignalSweep,
    StoredTraceSignal,
    candidate,
    run_signal_loop,
    run_signal_tick,
    signal_state,
    trace_signals,
)
from litellm.proxy.lens.sources import SourceReader
from litellm.rust_bridge.trace.generated.models import (
    ActivityAvailability,
    AgentRow,
    CountRow,
    ExecutionRow,
    LensAccessParams,
    LensContentParams,
    LensEvidenceParams,
    LensSampleParams,
    PartRow,
)
from litellm.types.decisions import DecisionsResponse
from litellm.types.decisions import NoulAnswer as DecisionsNoulAnswer

NOW: Final = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
CURRENT_CONFIG_KEY: Final = SignalConfig(model="decision").key()
_SIGNAL_STEPS: Final[TypeAdapter[tuple[SignalStep, ...]]] = TypeAdapter(tuple[SignalStep, ...])
_STORED_DATA: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def execution(identity: str, span_count: int = 1) -> Execution:
    return Execution(
        id=identity,
        source="traces",
        trace_id=identity,
        team_id="",
        name=identity,
        start_time="",
        span_count=span_count,
    )


def part(identity: str, content: str) -> PartRow:
    return PartRow(
        span_id=identity,
        parent_span_id="",
        name=identity,
        kind="agent",
        start_time="",
        end_time="",
        content=content,
        truncated=0,
    )


def stored_trace(
    config_key: str,
    *,
    trace_id: str = "trace",
    status: str = "classified",
    span_count: int = 1,
    claimed_until: datetime | None = None,
    classified_at: datetime | None = NOW - timedelta(minutes=10),
    scores: dict[str, float] | None = None,
    error: str = "",
) -> StoredTraceSignal:
    return StoredTraceSignal(
        trace_id=trace_id,
        trace_ref="",
        config_key=config_key,
        span_count=span_count,
        claimed_until=claimed_until,
        classified_at=classified_at,
        data=_STORED_DATA.validate_python(
            {
                "status": status,
                "scores": scores or {},
                "model": "decision",
                "error": error,
            }
        ),
    )


class SignalStorage:
    def __init__(
        self,
        executions: tuple[ExecutionRow, ...] = (),
        parts: tuple[PartRow, ...] = (),
    ) -> None:
        self.executions: Final = executions
        self.parts: Final = parts

    async def lens_availability(self, parameters: LensAccessParams) -> Sequence[ActivityAvailability]:
        return ()

    async def lens_agents(self, parameters: LensAccessParams) -> Sequence[AgentRow]:
        return ()

    async def lens_sample(self, parameters: LensSampleParams) -> Sequence[ExecutionRow]:
        return self.executions

    async def lens_content(self, parameters: LensContentParams) -> Sequence[PartRow]:
        return self.parts or (part(parameters.id, parameters.id),)

    async def lens_evidence(self, parameters: LensEvidenceParams) -> Sequence[CountRow]:
        return ()


class PagedSignalStorage(SignalStorage):
    def __init__(self, pages: tuple[tuple[PartRow, ...], ...]) -> None:
        super().__init__()
        self.pages: Final = pages

    async def lens_content(self, parameters: LensContentParams) -> Sequence[PartRow]:
        index: Final = int(parameters.cursor) if parameters.cursor else 0
        return self.pages[index]


class PagedSampleStorage(SignalStorage):
    def __init__(self, pages: tuple[tuple[ExecutionRow, ...], ...], initial_cursor: str = "") -> None:
        super().__init__()
        self.pages: Final = pages
        self.cursors: Final[asyncio.Queue[str]] = asyncio.Queue()
        self.page_by_cursor: Final = MappingProxyType(
            {
                initial_cursor: 0,
                **{page[-1].selection_key: index + 1 for index, page in enumerate(pages[:-1])},
            }
        )

    async def lens_sample(self, parameters: LensSampleParams) -> Sequence[ExecutionRow]:
        await self.cursors.put(parameters.after)
        index: Final = self.page_by_cursor[parameters.after]
        return self.pages[index]


class SignalDatabase:
    def __init__(
        self,
        config: SignalConfig | None,
        *,
        stored_rows: tuple[StoredTraceSignal, ...] = (),
        claim_result: bool = True,
    ) -> None:
        self.config: Final = config
        self.stored_rows: Final = stored_rows
        self.claim_result: Final = claim_result
        self.calls: Final[asyncio.Queue[str]] = asyncio.Queue()
        self.claims: Final[asyncio.Queue[str]] = asyncio.Queue()
        self.claim_args: Final[asyncio.Queue[tuple[object, ...]]] = asyncio.Queue()
        self.saved: Final[asyncio.Queue[tuple[object, ...]]] = asyncio.Queue()

    async def query_raw(self, query: str, *args: object) -> object:
        if '"LiteLLM_LensSignalConfig"' in query:
            return () if self.config is None else (Row(data=self.config.model_dump(mode="json")),)
        if query.startswith("SELECT jsonb_build_object"):
            payload: Final = args[0]
            assert isinstance(payload, str)
            requested: Final = TypeAdapter(tuple[TraceIdentity, ...]).validate_json(payload)
            identities: Final = tuple((trace.trace_id, trace.trace_ref) for trace in requested)
            return tuple(
                Row(data=stored.model_dump(mode="json"))
                for stored in self.stored_rows
                if (stored.trace_id, stored.trace_ref) in identities
            )
        if query.startswith('INSERT INTO "LiteLLM_LensTraceSignal"'):
            await self.claim_args.put(args)
            if not self.claim_result:
                return ()
            trace_id: Final = args[0]
            assert isinstance(trace_id, str)
            await self.claims.put(trace_id)
            return (Row(data={"trace_id": trace_id}),)
        raise AssertionError(f"Unexpected query: {query}")

    async def execute_raw(self, query: str, *args: object) -> int:
        await self.saved.put(args)
        return 1

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Database, None]:
        yield self


def saved_result(args: tuple[object, ...]) -> SignalData:
    payload: Final = args[1]
    assert isinstance(payload, str)
    return SignalData.model_validate_json(payload)


@pytest.mark.asyncio
async def test_signal_repository_reads_defaults_and_saves_the_global_config() -> None:
    database: Final = SignalDatabase(None)
    repository: Final = SignalRepository(database)
    updated: Final = SignalConfig(model="decision", threshold=0.7)

    assert await repository.get_config() == SignalConfig()
    await repository.save_config(updated)

    saved: Final = await database.saved.get()
    assert saved[0] == "global"
    assert isinstance(saved[1], str)
    assert SignalConfig.model_validate_json(saved[1]) == updated


@pytest.mark.asyncio
async def test_signal_repository_reads_rows_and_reports_a_lost_claim() -> None:
    config: Final = SignalConfig(model="decision")
    row: Final = stored_trace(config.key())
    database: Final = SignalDatabase(config, stored_rows=(row,), claim_result=False)
    repository: Final = SignalRepository(database)

    assert await repository.traces(()) == ()
    assert await repository.traces((TraceIdentity(trace_id="trace"),)) == (row,)
    assert not await repository.claim(execution("trace"), config, NOW + timedelta(minutes=5), NOW)


def test_signal_config_hashes_questions_but_not_threshold_or_display_name() -> None:
    config: Final = SignalConfig(model="decision")
    different_threshold: Final = config.model_copy(update={"threshold": 0.9})
    renamed: Final = config.model_copy(
        update={
            "signals": (
                config.signals[0].model_copy(update={"name": "Frustration"}),
                *config.signals[1:],
            )
        }
    )
    changed_question: Final = config.model_copy(
        update={
            "signals": (
                config.signals[0].model_copy(update={"question": "Does this user sound upset?"}),
                *config.signals[1:],
            )
        }
    )

    assert config.key() == different_threshold.key() == renamed.key()
    assert config.key() != changed_question.key()
    assert DEFAULT_SIGNALS == config.signals


def test_signal_config_rejects_duplicate_ids_and_non_finite_thresholds() -> None:
    duplicate: Final = Signal(id="same", name="First", question="Question one")
    with pytest.raises(ValidationError):
        SignalConfig(signals=(duplicate, duplicate))
    with pytest.raises(ValidationError):
        SignalConfig(threshold=float("nan"))


@pytest.mark.parametrize(
    "stored,trace_count,expected",
    (
        (None, 1, True),
        (stored_trace("old"), 1, True),
        (
            stored_trace(CURRENT_CONFIG_KEY, span_count=1, classified_at=NOW - timedelta(minutes=6)),
            2,
            True,
        ),
        (
            stored_trace(
                CURRENT_CONFIG_KEY,
                status="failed",
                classified_at=(NOW - timedelta(minutes=31)).replace(tzinfo=None),
            ),
            1,
            True,
        ),
        (stored_trace(CURRENT_CONFIG_KEY), 1, False),
        (
            stored_trace(CURRENT_CONFIG_KEY, span_count=1, classified_at=NOW - timedelta(minutes=2)),
            2,
            False,
        ),
        (
            stored_trace("old", claimed_until=NOW + timedelta(minutes=1)),
            1,
            False,
        ),
        (
            stored_trace(
                CURRENT_CONFIG_KEY,
                status="pending",
                claimed_until=(NOW + timedelta(minutes=1)).replace(tzinfo=None),
                classified_at=None,
            ),
            1,
            False,
        ),
        (
            stored_trace(
                CURRENT_CONFIG_KEY,
                status="pending",
                claimed_until=(NOW - timedelta(minutes=1)).replace(tzinfo=None),
                classified_at=None,
            ),
            1,
            True,
        ),
        (stored_trace(CURRENT_CONFIG_KEY, span_count=2), 1, False),
        (stored_trace(CURRENT_CONFIG_KEY, span_count=1, classified_at=None), 2, False),
    ),
)
def test_candidate_selection_respects_config_span_age_failure_age_and_claims(
    stored: StoredTraceSignal | None, trace_count: int, expected: bool
) -> None:
    config: Final = SignalConfig(model="decision")
    assert candidate(execution("trace", trace_count), stored, config.key(), NOW) is expected


@pytest.mark.asyncio
async def test_classifier_sends_noul_questions_and_keeps_every_signal_score() -> None:
    config: Final = SignalConfig(model="decision")
    run: Final = execution("trace")
    storage: Final = SignalStorage(parts=(part("agent", "user asks for a result"),))

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        assert model == "decision"
        assert state == {
            "task": SIGNAL_TASK,
            "steps": ({"kind": "agent", "name": "agent", "content": "user asks for a result"},),
        }
        assert _STORED_DATA.validate_json(json.dumps(state)) == {
            "task": SIGNAL_TASK,
            "steps": [{"kind": "agent", "name": "agent", "content": "user asks for a result"}],
        }
        expected_questions: Final = {
            signal.id: {"type": "noul", "instructions": signal.question} for signal in config.signals
        }
        assert questions == expected_questions
        assert _STORED_DATA.validate_json(json.dumps(questions)) == expected_questions
        assert timeout == 60
        assert metadata == {"tags": ["litellm-lens-signals"]}
        return DecisionsResponse(
            answers={
                "user_frustration": DecisionsNoulAnswer(type="noul", noul=0.9),
                "missing_capability": DecisionsNoulAnswer(type="noul", noul=0.6),
                "repeated_request": DecisionsNoulAnswer(type="noul", noul=0.2),
                "unknown": DecisionsNoulAnswer(type="noul", noul=1.0),
            }
        )

    attempt: Final = await SignalClassifier(SourceReader(storage), decide, lambda: NOW).classify(
        Scope(all_teams=True), run, config
    )

    assert attempt == SignalAttempt(
        status="classified",
        scores={"user_frustration": 0.9, "missing_capability": 0.6, "repeated_request": 0.2},
        model="decision",
    )


@pytest.mark.asyncio
async def test_missing_noul_answer_fails_while_unknown_and_non_noul_answers_are_ignored() -> None:
    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {
            "answers": {
                "user_frustration": {"type": "noul", "noul": 0.9},
                "missing_capability": {"type": "choice", "choice": "yes"},
                "unknown": {"type": "noul", "noul": 1.0},
            }
        }

    attempt: Final = await SignalClassifier(
        SourceReader(SignalStorage(parts=(part("agent", "content"),))),
        decide,
        lambda: NOW,
    ).classify(Scope(all_teams=True), execution("trace"), SignalConfig(model="decision"))

    assert attempt.status == "failed"
    assert attempt.scores == {"user_frustration": 0.9}
    assert attempt.error == "Decisions response omitted a configured noul answer"


@pytest.mark.asyncio
async def test_classifier_turns_decisions_errors_into_failed_attempts() -> None:
    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        raise RuntimeError("decisions unavailable")

    attempt: Final = await SignalClassifier(
        SourceReader(SignalStorage(parts=(part("agent", "content"),))),
        decide,
        lambda: NOW,
    ).classify(Scope(all_teams=True), execution("trace"), SignalConfig(model="decision"))

    assert attempt == SignalAttempt(status="failed", model="decision", error="decisions unavailable")


def test_signal_flags_use_current_threshold_and_current_display_name() -> None:
    config: Final = SignalConfig(model="decision")
    row: Final = stored_trace(
        config.key(),
        scores={"user_frustration": 0.91, "missing_capability": 0.67, "repeated_request": 0.49},
    )
    high_threshold: Final = config.model_copy(
        update={
            "threshold": 0.9,
            "signals": (
                config.signals[0].model_copy(update={"name": "Frustrated user"}),
                *config.signals[1:],
            ),
        }
    )
    trace: Final = TraceIdentity(trace_id="trace")
    lower: Final = trace_signals(trace, row, config)
    higher: Final = trace_signals(trace, row, high_threshold)

    assert config.key() == high_threshold.key()
    assert tuple((flag.signal_id, flag.score) for flag in lower.flags) == (
        ("user_frustration", 0.91),
        ("missing_capability", 0.67),
    )
    assert tuple((flag.signal_id, flag.name, flag.score) for flag in higher.flags) == (
        ("user_frustration", "Frustrated user", 0.91),
    )
    assert not candidate(execution("trace"), row, high_threshold.key(), NOW)


def test_signal_flags_report_stored_errors_even_when_the_status_is_classified() -> None:
    config: Final = SignalConfig(model="decision")
    row: Final = stored_trace(config.key(), status="classified", error="classification failed")

    result: Final = trace_signals(TraceIdentity(trace_id="trace"), row, config)

    assert result.status == "failed"
    assert result.model == "decision"
    assert result.classified_at == row.classified_at


@pytest.mark.asyncio
async def test_signal_state_caps_content_to_head_and_tail_with_omitted_step() -> None:
    parts: Final = tuple(part(str(index), chr(97 + index) * 2000) for index in range(30))
    state: Final = await signal_state(
        SourceReader(SignalStorage(parts=parts)),
        Scope(all_teams=True),
        execution("trace"),
    )
    steps_value: Final = state["steps"]
    assert isinstance(steps_value, tuple)
    steps: Final = _SIGNAL_STEPS.validate_python(steps_value)
    head: Final = steps[:8]
    marker: Final = steps[8]
    tail: Final = steps[9:]

    assert state["task"] == SIGNAL_TASK
    assert sum(len(step.content) for step in head) == 15000
    assert sum(len(step.content) for step in tail) == 25000
    assert head[0].content == "a" * 2000
    assert head[-1].content == "h" * 1000
    assert marker == SignalStep(kind="omitted", name="", content="9 steps omitted")
    assert tail[0].content == "r" * 1000
    assert tail[-1].content == "~" * 2000


@pytest.mark.asyncio
async def test_signal_state_limits_content_pages_and_part_sizes() -> None:
    pages: Final = tuple(
        tuple(part(f"page-{page}-{index}", "x" * 2501 if index == 0 else "x") for index in range(39))
        + (part(str(page + 1), "x"),)
        for page in range(4)
    )
    state: Final = await signal_state(
        SourceReader(PagedSignalStorage(pages)),
        Scope(all_teams=True),
        execution("trace"),
    )
    steps: Final = _SIGNAL_STEPS.validate_python(state["steps"])

    assert len(steps) == 120
    assert steps[0].content.startswith("x" * 800)
    assert "[... 501 characters omitted ...]" in steps[0].content
    assert steps[0].content.endswith("x" * 1200)
    assert steps[-1].name == "3"
    assert all(not step.name.startswith("page-3-") for step in steps)

    small_state: Final = await signal_state(
        SourceReader(SignalStorage(parts=(part("small", "ok"),))),
        Scope(all_teams=True),
        execution("trace"),
    )
    small_steps: Final = _SIGNAL_STEPS.validate_python(small_state["steps"])
    assert small_steps == (SignalStep(kind="agent", name="small", content="ok"),)


@pytest.mark.asyncio
async def test_signal_state_part_excerpt_preserves_the_output_tail() -> None:
    content: Final = "I" * 5000 + "OUTPUT: refused"
    state: Final = await signal_state(
        SourceReader(SignalStorage(parts=(part("result", content),))),
        Scope(all_teams=True),
        execution("trace"),
    )
    steps: Final = _SIGNAL_STEPS.validate_python(state["steps"])
    excerpt: Final = steps[0].content
    marker: Final = "\n[... 3015 characters omitted ...]\n"

    assert marker in excerpt
    assert excerpt.endswith("OUTPUT: refused")
    assert len(excerpt) == 800 + len(marker) + 1200


@pytest.mark.asyncio
async def test_signal_tick_classifies_at_most_50_traces_and_persists_scores() -> None:
    config: Final = SignalConfig(model="decision")
    executions: Final = tuple(
        ExecutionRow(
            source="traces",
            trace_id=f"trace-{index}",
            team_id="",
            name=f"trace-{index}",
            start_time="",
            span_count=1,
            root_seen=1,
            eligible=60,
            selected=60,
            selection_key=f"cursor-{index}",
        )
        for index in range(60)
    )
    storage: Final = SignalStorage(executions=executions)
    database: Final = SignalDatabase(config)
    repository: Final = SignalRepository(database)

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        steps: Final = TypeAdapter(tuple[SignalStep, ...]).validate_python(state["steps"])
        await database.calls.put(steps[0].name)
        return {
            "answers": {
                "user_frustration": {"type": "noul", "noul": 0.9},
                "missing_capability": {"type": "noul", "noul": 0.6},
                "repeated_request": {"type": "noul", "noul": 0.2},
            }
        }

    await run_signal_tick(storage, repository, decide, lambda: NOW)
    classified: Final = tuple(database.saved.get_nowait() for _ in range(database.saved.qsize()))
    traces: Final = tuple(database.calls.get_nowait() for _ in range(database.calls.qsize()))
    saved_data: Final = tuple(saved_result(args) for args in classified)

    assert len(classified) == 50
    assert frozenset(traces) == frozenset(f"trace-{index}" for index in range(50))
    assert (
        saved_data
        == (
            SignalData(
                status="classified",
                scores={
                    "user_frustration": 0.9,
                    "missing_capability": 0.6,
                    "repeated_request": 0.2,
                },
                model="decision",
                error="",
            ),
        )
        * 50
    )


@pytest.mark.asyncio
async def test_signal_tick_claims_with_worker_start_time_and_skips_lost_claims() -> None:
    config: Final = SignalConfig(model="decision")
    executions: Final = tuple(
        ExecutionRow(
            source="traces",
            trace_id=f"trace-{index}",
            team_id="",
            name=f"trace-{index}",
            start_time="",
            span_count=1,
            root_seen=1,
            eligible=2,
            selected=2,
            selection_key=f"cursor-{index}",
        )
        for index in range(2)
    )
    database: Final = SignalDatabase(config, claim_result=False)
    repository: Final = SignalRepository(database)

    class AdvancingClock:
        def __init__(self) -> None:
            self.values: Final = tuple(NOW + timedelta(minutes=index) for index in range(3))
            self.index: int = 0

        def __call__(self) -> datetime:
            value: Final = self.values[self.index]
            self.index += 1
            return value

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {"answers": {}}

    await run_signal_tick(SignalStorage(executions=executions), repository, decide, AdvancingClock())

    claims: Final = tuple(database.claim_args.get_nowait() for _ in range(database.claim_args.qsize()))

    def claim_times(args: tuple[object, ...]) -> tuple[datetime, datetime]:
        claimed_until: Final = args[4]
        claimed_at: Final = args[6]
        assert isinstance(claimed_until, datetime)
        assert isinstance(claimed_at, datetime)
        return claimed_until, claimed_at

    times: Final = tuple(claim_times(claim) for claim in claims)
    assert database.calls.empty()
    assert database.saved.empty()
    assert all(claimed_until == claimed_at + SIGNAL_CLAIM_LEASE for claimed_until, claimed_at in times)
    assert all(claimed_at != NOW for _, claimed_at in times)


@pytest.mark.asyncio
async def test_signal_tick_resumes_after_ten_pages_and_resets_after_a_short_page() -> None:
    config: Final = SignalConfig(model="decision")

    def sample_page(page: int) -> tuple[ExecutionRow, ...]:
        return tuple(
            ExecutionRow(
                source="traces",
                trace_id=f"trace-{page}-{index}",
                team_id="",
                name=f"trace-{page}-{index}",
                start_time="",
                span_count=1,
                root_seen=1,
                eligible=2500,
                selected=2500,
                selection_key=f"page-{page}-{index}",
            )
            for index in range(100)
        )

    pages: Final = tuple(sample_page(page) for page in range(25))
    all_rows: Final = tuple(chain.from_iterable(pages))
    stored_rows: Final = tuple(stored_trace(CURRENT_CONFIG_KEY, trace_id=row.trace_id) for row in all_rows)
    storage: Final = PagedSampleStorage(pages)
    database: Final = SignalDatabase(config, stored_rows=stored_rows)
    repository: Final = SignalRepository(database)

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {"answers": {}}

    first_cursor: Final = (await run_signal_tick(storage, repository, decide, lambda: NOW)).cursor
    first_calls: Final = tuple(storage.cursors.get_nowait() for _ in range(storage.cursors.qsize()))
    second_cursor: Final = (
        await run_signal_tick(
            storage,
            repository,
            decide,
            lambda: NOW,
            cursor=first_cursor,
        )
    ).cursor
    second_calls: Final = tuple(storage.cursors.get_nowait() for _ in range(storage.cursors.qsize()))

    assert len(first_calls) == SIGNAL_MAX_SCAN_PAGES
    assert first_cursor
    assert len(second_calls) == SIGNAL_MAX_SCAN_PAGES
    assert second_calls[0] == first_cursor
    assert second_cursor

    short_storage: Final = PagedSampleStorage((pages[0][:50],))
    short_database: Final = SignalDatabase(config, stored_rows=stored_rows[:50])
    short_cursor: Final = (
        await run_signal_tick(
            short_storage,
            SignalRepository(short_database),
            decide,
            lambda: NOW,
        )
    ).cursor
    assert short_cursor == ""


@pytest.mark.asyncio
async def test_signal_tick_resumes_a_partially_consumed_page() -> None:
    config: Final = SignalConfig(model="decision")
    page: Final = tuple(
        ExecutionRow(
            source="traces",
            trace_id=f"trace-{index}",
            team_id="",
            name=f"trace-{index}",
            start_time="",
            span_count=1,
            root_seen=1,
            eligible=100,
            selected=100,
            selection_key=f"cursor-{index}",
        )
        for index in range(100)
    )
    initial_rows: Final = tuple(stored_trace(CURRENT_CONFIG_KEY, trace_id=f"trace-{index}") for index in range(20))
    resume_cursor: Final = "resume-page"
    storage: Final = PagedSampleStorage((page, ()), initial_cursor=resume_cursor)

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {
            "answers": {
                "user_frustration": {"type": "noul", "noul": 0.9},
                "missing_capability": {"type": "noul", "noul": 0.6},
                "repeated_request": {"type": "noul", "noul": 0.2},
            }
        }

    first_database: Final = SignalDatabase(config, stored_rows=initial_rows)
    first_cursor: Final = (
        await run_signal_tick(
            storage,
            SignalRepository(first_database),
            decide,
            lambda: NOW,
            cursor=resume_cursor,
        )
    ).cursor
    first_claims: Final = tuple(first_database.claims.get_nowait() for _ in range(first_database.claims.qsize()))

    classified_first_rows: Final = tuple(
        stored_trace(CURRENT_CONFIG_KEY, trace_id=trace_id) for trace_id in first_claims
    )
    second_database: Final = SignalDatabase(config, stored_rows=(*initial_rows, *classified_first_rows))
    second_cursor: Final = (
        await run_signal_tick(
            storage,
            SignalRepository(second_database),
            decide,
            lambda: NOW,
            cursor=first_cursor,
        )
    ).cursor
    second_claims: Final = tuple(second_database.claims.get_nowait() for _ in range(second_database.claims.qsize()))
    sample_cursors: Final = tuple(storage.cursors.get_nowait() for _ in range(storage.cursors.qsize()))
    expected_eligible: Final = frozenset(f"trace-{index}" for index in range(20, 100))

    assert first_cursor == resume_cursor
    assert second_cursor == ""
    assert len(first_claims) == 50
    assert len(second_claims) == 30
    assert frozenset(first_claims).isdisjoint(second_claims)
    assert frozenset(first_claims) | frozenset(second_claims) == expected_eligible
    assert sample_cursors == (resume_cursor, resume_cursor, page[-1].selection_key)


@pytest.mark.asyncio
async def test_signal_tick_skips_claims_and_writes_when_router_is_not_ready() -> None:
    config: Final = SignalConfig(model="decision")
    storage: Final = SignalStorage(
        executions=(
            ExecutionRow(
                source="traces",
                trace_id="trace",
                team_id="",
                name="trace",
                start_time="",
                span_count=1,
                root_seen=1,
                eligible=1,
                selected=1,
                selection_key="cursor",
            ),
        )
    )
    database: Final = SignalDatabase(config)

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {"answers": {}}

    await run_signal_tick(
        storage,
        SignalRepository(database),
        decide,
        lambda: NOW,
        router_ready=lambda: False,
    )

    assert database.claims.empty()
    assert database.saved.empty()


@pytest.mark.asyncio
async def test_signal_tick_skips_missing_dependencies_and_disabled_configs() -> None:
    storage: Final = SignalStorage()

    await run_signal_tick(storage, None, None, lambda: NOW)

    database: Final = SignalDatabase(SignalConfig())

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        raise AssertionError("disabled signal config should not call Decisions")

    await run_signal_tick(storage, SignalRepository(database), decide, lambda: NOW)
    assert database.claims.empty()
    assert database.saved.empty()


class FailingStoreDatabase(SignalDatabase):
    async def execute_raw(self, query: str, *args: object) -> int:
        raise RuntimeError("store unavailable")


@pytest.mark.asyncio
async def test_signal_tick_continues_when_storing_a_result_fails() -> None:
    config: Final = SignalConfig(model="decision")
    execution_row: Final = ExecutionRow(
        source="traces",
        trace_id="trace",
        team_id="",
        name="trace",
        start_time="",
        span_count=1,
        root_seen=1,
        eligible=1,
        selected=1,
        selection_key="cursor",
    )
    database: Final = FailingStoreDatabase(config)

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {
            "answers": {
                "user_frustration": {"type": "noul", "noul": 0.9},
                "missing_capability": {"type": "noul", "noul": 0.6},
                "repeated_request": {"type": "noul", "noul": 0.2},
            }
        }

    await run_signal_tick(
        SignalStorage(executions=(execution_row,)),
        SignalRepository(database),
        decide,
        lambda: NOW,
    )

    assert await database.claims.get() == "trace"
    assert database.saved.empty()


class FailingSignalRepository:
    def __init__(self) -> None:
        self.started: Final = asyncio.Event()

    async def get_config(self) -> SignalConfig:
        self.started.set()
        await asyncio.sleep(0)
        raise RuntimeError("tick failed")


@pytest.mark.asyncio
async def test_signal_loop_continues_after_a_tick_error() -> None:
    repository: Final = FailingSignalRepository()

    async def decide(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return {"answers": {}}

    task: Final = asyncio.create_task(run_signal_loop(SignalStorage(), repository, decide, lambda: NOW))
    await repository.started.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_proxy_signal_call_resolves_the_current_router(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    async def first_decisions(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return "first"

    async def second_decisions(
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object:
        return "second"

    async def call_current_router() -> object:
        return await proxy_server._call_current_lens_signal_router(
            model="decision",
            state={"task": "task"},
            questions={},
            timeout=60,
            metadata={"tags": ["test"]},
        )

    monkeypatch.setattr(proxy_server, "llm_router", SimpleNamespace(adecisions=first_decisions))
    assert await call_current_router() == "first"

    monkeypatch.setattr(proxy_server, "llm_router", SimpleNamespace(adecisions=second_decisions))
    assert await call_current_router() == "second"

    monkeypatch.setattr(proxy_server, "llm_router", None)
    with pytest.raises(RuntimeError, match="router is not initialized"):
        await call_current_router()


class RecordingSampleStorage(PagedSampleStorage):
    def __init__(self, pages: tuple[tuple[ExecutionRow, ...], ...]) -> None:
        super().__init__(pages)
        self.windows: Final[asyncio.Queue[tuple[int, int]]] = asyncio.Queue()

    async def lens_sample(self, parameters: LensSampleParams) -> Sequence[ExecutionRow]:
        await self.windows.put((parameters.start, parameters.end))
        return await super().lens_sample(parameters)


def sample_rows(prefix: str, count: int) -> tuple[ExecutionRow, ...]:
    return tuple(
        ExecutionRow(
            source="traces",
            trace_id=f"{prefix}-{index}",
            team_id="",
            name=f"{prefix}-{index}",
            start_time="",
            span_count=1,
            root_seen=1,
            eligible=count,
            selected=count,
            selection_key=f"{prefix}-{index}",
        )
        for index in range(count)
    )


async def no_answers(
    *,
    model: str,
    state: DecisionState,
    questions: DecisionQuestions,
    timeout: float,
    metadata: Mapping[str, object],
) -> object:
    return {"answers": {}}


def drained(queue: "asyncio.Queue[tuple[int, int]]") -> tuple[tuple[int, int], ...]:
    return tuple(queue.get_nowait() for _ in range(queue.qsize()))


@pytest.mark.asyncio
async def test_live_sweep_reads_one_page_of_recently_finished_traces() -> None:
    pages: Final = (sample_rows("a", 100), sample_rows("b", 100), ())
    stored_rows: Final = tuple(
        stored_trace(CURRENT_CONFIG_KEY, trace_id=row.trace_id) for row in chain.from_iterable(pages)
    )
    live_storage: Final = RecordingSampleStorage(pages)
    backlog_storage: Final = RecordingSampleStorage(pages)
    repository: Final = SignalRepository(SignalDatabase(SignalConfig(model="decision"), stored_rows=stored_rows))

    live_tick: Final = await run_signal_tick(live_storage, repository, no_answers, lambda: NOW, sweep=SIGNAL_LIVE_SWEEP)
    await run_signal_tick(backlog_storage, repository, no_answers, lambda: NOW, sweep=SIGNAL_BACKLOG_SWEEP)
    live_windows: Final = drained(live_storage.windows)
    backlog_windows: Final = drained(backlog_storage.windows)
    now_ms: Final = int(NOW.timestamp() * 1000)

    assert len(live_windows) == 1
    assert live_tick.cursor == pages[0][-1].selection_key
    assert live_tick.claimed == 0
    assert backlog_windows[0][0] < live_windows[0][0] < live_windows[0][1] < now_ms
    assert live_windows[0][1] == backlog_windows[0][1]
    assert now_ms - live_windows[0][1] <= 30_000, "a finished trace should be visible to the sweep within seconds"


@pytest.mark.asyncio
async def test_signal_loop_drains_a_backlog_without_waiting_for_the_interval() -> None:
    storage: Final = SignalStorage(executions=sample_rows("trace", SIGNAL_MAX_PER_TICK + 10))
    database: Final = SignalDatabase(SignalConfig(model="decision"))
    hour_long_sweep: Final = SignalSweep(lookback=timedelta(minutes=15), interval_seconds=3600, max_pages=1)

    task: Final = asyncio.create_task(
        run_signal_loop(storage, SignalRepository(database), no_answers, lambda: NOW, sweep=hour_long_sweep)
    )
    claims: Final = tuple([await asyncio.wait_for(database.claims.get(), 1) for _ in range(SIGNAL_MAX_PER_TICK + 1)])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(claims) == SIGNAL_MAX_PER_TICK + 1
