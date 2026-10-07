import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy.lens.models import Execution, Scope, TraceIdentity
from litellm.proxy.lens.repository import Database, Row
from litellm.proxy.lens.signal_repository import SignalRepository
from litellm.proxy.lens.signals import (
    DEFAULT_SIGNALS,
    SIGNAL_TASK,
    DecisionQuestions,
    DecisionState,
    Signal,
    SignalAttempt,
    SignalClassifier,
    SignalConfig,
    SignalData,
    SignalStep,
    StoredTraceSignal,
    candidate,
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
    status: str = "classified",
    span_count: int = 1,
    claimed_until: datetime | None = None,
    classified_at: datetime | None = NOW - timedelta(minutes=10),
    scores: dict[str, float] | None = None,
    error: str = "",
) -> StoredTraceSignal:
    return StoredTraceSignal(
        trace_id="trace",
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


class SignalDatabase:
    def __init__(self, config: SignalConfig) -> None:
        self.config: Final = config
        self.calls: Final[asyncio.Queue[str]] = asyncio.Queue()
        self.saved: Final[asyncio.Queue[tuple[object, ...]]] = asyncio.Queue()

    async def query_raw(self, query: str, *args: object) -> object:
        if '"LiteLLM_LensSignalConfig"' in query:
            return (Row(data=self.config.model_dump(mode="json")),)
        if query.startswith('SELECT jsonb_build_object'):
            return ()
        if query.startswith('INSERT INTO "LiteLLM_LensTraceSignal"'):
            trace_id: Final = args[0]
            assert isinstance(trace_id, str)
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
    assert saved_data == (
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
    ) * 50
