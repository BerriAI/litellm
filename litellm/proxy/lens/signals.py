import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import accumulate
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

from litellm.integrations.clickhouse.context import lens_analysis
from litellm.litellm_core_utils.initialize_dynamic_callback_params import inherit_message_logging_privacy
from litellm.litellm_core_utils.secret_redaction import redact_internal_details
from litellm.proxy.lens.models import ActivitySelection, Execution, Record, Scope, TraceIdentity
from litellm.proxy.lens.sources import SourceReader, Storage

SIGNAL_SETTLE: Final = timedelta(seconds=15)
SIGNAL_PAGE_SIZE: Final = 100
SIGNAL_MAX_PER_TICK: Final = 50
SIGNAL_CONCURRENCY: Final = 8
SIGNAL_CLAIM_LEASE: Final = timedelta(minutes=5)
SIGNAL_RECLASSIFY_AFTER: Final = timedelta(minutes=5)
SIGNAL_RETRY_FAILED_AFTER: Final = timedelta(minutes=30)
SIGNAL_MAX_CONTENT_PAGES: Final = 3
SIGNAL_PART_MAX_CHARS: Final = 2000
SIGNAL_PART_HEAD_CHARS: Final = 800
SIGNAL_PART_TAIL_CHARS: Final = 1200
SIGNAL_TRANSCRIPT_MAX_CHARS: Final = 40000
SIGNAL_TRANSCRIPT_HEAD_CHARS: Final = 15000
SIGNAL_TRANSCRIPT_TAIL_CHARS: Final = 25000
SIGNAL_MAX_SCAN_PAGES: Final = 10


@dataclass(frozen=True, slots=True)
class SignalSweep:
    lookback: timedelta
    interval_seconds: float
    max_pages: int


SIGNAL_LIVE_SWEEP: Final = SignalSweep(lookback=timedelta(minutes=15), interval_seconds=2, max_pages=1)
SIGNAL_BACKLOG_SWEEP: Final = SignalSweep(
    lookback=timedelta(hours=24), interval_seconds=60, max_pages=SIGNAL_MAX_SCAN_PAGES
)
SIGNAL_TASK: Final = (
    "An AI agent run recorded as a trace. Judge only what the user and the agent said and did in these steps."
)


class Signal(Record):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    name: str = Field(min_length=1, max_length=60)
    question: str = Field(min_length=3, max_length=500)


DEFAULT_SIGNALS: Final[tuple[Signal, ...]] = (
    Signal(
        id="user_frustration",
        name="User frustration",
        question=(
            "Does the user show frustration, annoyance or dissatisfaction with the agent in this run, for example "
            "complaints, irritated corrections, all caps, profanity, or giving up on the task?"
        ),
    ),
    Signal(
        id="missing_capability",
        name="Missing capability",
        question=(
            "Does the user ask for something the agent cannot do in this run, so that the agent refuses, says it "
            "lacks a tool, permission, integration or data source, or fails because the capability does not exist?"
        ),
    ),
    Signal(
        id="repeated_request",
        name="Repeated request",
        question=(
            "Does the user ask for the same thing more than once in this run, usually because the agent did not "
            "deliver it the first time?"
        ),
    ),
)


class SignalConfig(Record):
    model: str = ""
    threshold: float = Field(default=0.5, ge=0.05, le=0.95, allow_inf_nan=False)
    signals: tuple[Signal, ...] = DEFAULT_SIGNALS

    @model_validator(mode="after")
    def validate_signals(self) -> "SignalConfig":
        if len(self.signals) > 20:
            raise ValueError("A maximum of 20 signals is allowed")
        if len(frozenset(signal.id for signal in self.signals)) != len(self.signals):
            raise ValueError("Signal IDs must be unique")
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.model) and bool(self.signals)

    def key(self) -> str:
        payload: Final = json.dumps(
            {
                "model": self.model,
                "signals": tuple({"id": signal.id, "question": signal.question} for signal in self.signals),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


Score: TypeAlias = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class SignalFlag(Record):
    signal_id: str
    name: str
    score: Score


class TraceSignals(TraceIdentity):
    status: Literal["unclassified", "pending", "classified", "failed"]
    flags: tuple[SignalFlag, ...] = ()
    model: str = ""
    classified_at: datetime | None = None


class SignalStep(Record):
    kind: str
    name: str
    content: str


class SignalData(Record):
    model_config = ConfigDict(extra="ignore")

    status: Literal["pending", "classified", "failed"] = "pending"
    scores: Mapping[str, Score] = Field(default_factory=lambda: MappingProxyType({}))
    model: str = ""
    error: str = ""


class SignalAttempt(Record):
    status: Literal["classified", "failed"]
    scores: Mapping[str, Score] = Field(default_factory=lambda: MappingProxyType({}))
    model: str
    error: str = ""


class StoredTraceSignal(Record):
    trace_id: str
    trace_ref: str = ""
    config_key: str
    span_count: int
    claimed_until: datetime | None = None
    classified_at: datetime | None = None
    data: JsonValue

    @field_validator("claimed_until", "classified_at")
    @classmethod
    def normalize_database_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class NoulAnswer(Record):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False, from_attributes=True)

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, allow_inf_nan=False)


class DecisionsOutput(Record):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    answers: Mapping[str, object]


DecisionState: TypeAlias = Mapping[str, object]
DecisionQuestions: TypeAlias = Mapping[str, Mapping[str, str]]
Clock: TypeAlias = Callable[[], datetime]
RouterReady: TypeAlias = Callable[[], bool]


class DecisionsCall(Protocol):
    async def __call__(
        self,
        *,
        model: str,
        state: DecisionState,
        questions: DecisionQuestions,
        timeout: float,
        metadata: Mapping[str, object],
    ) -> object: ...


class SignalRepositoryProtocol(Protocol):
    async def get_config(self) -> SignalConfig: ...
    async def traces(self, identities: tuple[TraceIdentity, ...]) -> tuple[StoredTraceSignal, ...]: ...
    async def claim(
        self,
        execution: Execution,
        config: SignalConfig,
        claimed_until: datetime,
        now: datetime,
    ) -> bool: ...
    async def store(
        self,
        execution: Execution,
        config: SignalConfig,
        claimed_until: datetime,
        classified_at: datetime,
        attempt: SignalAttempt,
    ) -> None: ...


def signal_identity(trace: TraceIdentity | StoredTraceSignal | Execution) -> tuple[str, str]:
    return trace.trace_id, trace.trace_ref


def candidate(
    trace: Execution,
    existing: StoredTraceSignal | None,
    config_key: str,
    now: datetime,
) -> bool:
    if existing is None:
        return True
    if existing.claimed_until is not None and existing.claimed_until > now:
        return False
    if existing.config_key != config_key:
        return True
    status: Final = existing.data.get("status") if isinstance(existing.data, dict) else ""
    if status == "pending":
        return existing.claimed_until is not None and existing.claimed_until <= now
    if existing.span_count > trace.span_count:
        return False
    if existing.span_count < trace.span_count:
        return existing.classified_at is not None and existing.classified_at < now - SIGNAL_RECLASSIFY_AFTER
    return (
        status == "failed"
        and existing.classified_at is not None
        and existing.classified_at < now - SIGNAL_RETRY_FAILED_AFTER
    )


def _take_head(steps: tuple[SignalStep, ...], remaining: int) -> tuple[SignalStep, ...]:
    if remaining <= 0:
        return ()
    cumulative_lengths: Final = tuple(accumulate(len(step.content) for step in steps))
    boundary: Final = next((index for index, total in enumerate(cumulative_lengths) if total >= remaining), None)
    if boundary is None:
        return steps
    preceding: Final = steps[:boundary]
    last: Final = steps[boundary]
    used: Final = cumulative_lengths[boundary - 1] if boundary > 0 else 0
    last_length: Final = remaining - used
    return (
        *preceding,
        last
        if last_length == len(last.content)
        else last.model_copy(update=MappingProxyType({"content": last.content[:last_length]})),
    )


def _take_tail(steps: tuple[SignalStep, ...], remaining: int) -> tuple[SignalStep, ...]:
    if remaining <= 0:
        return ()
    reversed_steps: Final = tuple(reversed(steps))
    cumulative_lengths: Final = tuple(accumulate(len(step.content) for step in reversed_steps))
    boundary: Final = next((index for index, total in enumerate(cumulative_lengths) if total >= remaining), None)
    if boundary is None:
        return steps
    preceding: Final = reversed_steps[:boundary]
    last: Final = reversed_steps[boundary]
    used: Final = cumulative_lengths[boundary - 1] if boundary > 0 else 0
    last_length: Final = remaining - used
    selected: Final = (
        *preceding,
        last
        if last_length == len(last.content)
        else last.model_copy(update=MappingProxyType({"content": last.content[-last_length:]})),
    )
    return tuple(reversed(selected))


def _bounded_steps(steps: tuple[SignalStep, ...]) -> tuple[SignalStep, ...]:
    if sum(len(step.content) for step in steps) <= SIGNAL_TRANSCRIPT_MAX_CHARS:
        return steps
    head: Final = _take_head(steps, SIGNAL_TRANSCRIPT_HEAD_CHARS)
    tail: Final = _take_tail(steps, SIGNAL_TRANSCRIPT_TAIL_CHARS)
    omitted_count: Final = len(steps) - len(head) - len(tail)
    marker: Final = SignalStep(kind="omitted", name="", content=f"{omitted_count} steps omitted")
    return (*head, marker, *tail)


def _part_excerpt(content: str) -> str:
    if len(content) <= SIGNAL_PART_MAX_CHARS:
        return content
    omitted: Final = len(content) - SIGNAL_PART_MAX_CHARS
    marker: Final = f"\n[... {omitted} characters omitted ...]\n"
    return f"{content[:SIGNAL_PART_HEAD_CHARS]}{marker}{content[-SIGNAL_PART_TAIL_CHARS:]}"


async def _content_pages(
    reader: SourceReader,
    scope: Scope,
    execution: Execution,
    cursor: str,
    pages_left: int,
) -> tuple[SignalStep, ...]:
    if pages_left == 0:
        return ()
    content: Final = await reader.content(scope, execution, cursor)
    current: Final = tuple(
        SignalStep(kind=part.kind, name=part.name, content=_part_excerpt(part.content)) for part in content.parts
    )
    rest: Final = (
        await _content_pages(reader, scope, execution, content.next_cursor, pages_left - 1)
        if content.next_cursor is not None
        else ()
    )
    return (*current, *rest)


async def signal_state(reader: SourceReader, scope: Scope, execution: Execution) -> DecisionState:
    steps: Final = _bounded_steps(await _content_pages(reader, scope, execution, "", SIGNAL_MAX_CONTENT_PAGES))
    return {
        "task": SIGNAL_TASK,
        "steps": tuple(step.model_dump(mode="json") for step in steps),
    }


def _noul_score(value: object) -> float | None:
    try:
        return NoulAnswer.model_validate(value).noul
    except ValidationError:
        return None


class SignalClassifier:
    def __init__(self, reader: SourceReader, completion: DecisionsCall, clock: Clock) -> None:
        self.reader: Final = reader
        self.completion: Final = completion
        self.clock: Final = clock

    async def classify(self, scope: Scope, execution: Execution, config: SignalConfig) -> SignalAttempt:
        try:
            state: Final = await signal_state(self.reader, scope, execution)
            questions: Final = {
                signal.id: {"type": "noul", "instructions": signal.question} for signal in config.signals
            }
            with lens_analysis(), inherit_message_logging_privacy(True):
                response: Final = await self.completion(
                    model=config.model,
                    state=state,
                    questions=questions,
                    timeout=60,
                    metadata={"tags": ["litellm-lens-signals"]},
                )
            output: Final = DecisionsOutput.model_validate(response)
            scores: Final = MappingProxyType(
                {
                    signal.id: score
                    for signal in config.signals
                    if (score := _noul_score(output.answers.get(signal.id))) is not None
                }
            )
            if len(scores) != len(config.signals):
                return SignalAttempt(
                    status="failed",
                    scores=scores,
                    model=config.model,
                    error="Decisions response omitted a configured noul answer",
                )
            return SignalAttempt(status="classified", scores=scores, model=config.model)
        except Exception as error:
            detail: Final = redact_internal_details(str(error))[:300]
            return SignalAttempt(status="failed", model=config.model, error=detail)


def trace_signals(
    trace: TraceIdentity,
    existing: StoredTraceSignal | None,
    config: SignalConfig,
) -> TraceSignals:
    if existing is None or existing.config_key != config.key():
        return TraceSignals(trace_id=trace.trace_id, trace_ref=trace.trace_ref, status="unclassified")
    data: Final = SignalData.model_validate(existing.data)
    if data.status == "pending":
        return TraceSignals(
            trace_id=trace.trace_id,
            trace_ref=trace.trace_ref,
            status="pending",
            model=data.model,
        )
    if data.status == "failed" or data.error:
        return TraceSignals(
            trace_id=trace.trace_id,
            trace_ref=trace.trace_ref,
            status="failed",
            model=data.model,
            classified_at=existing.classified_at,
        )
    flags: Final = tuple(
        sorted(
            (
                SignalFlag(signal_id=signal.id, name=signal.name, score=data.scores[signal.id])
                for signal in config.signals
                if signal.id in data.scores and data.scores[signal.id] >= config.threshold
            ),
            key=lambda flag: flag.score,
            reverse=True,
        )
    )
    return TraceSignals(
        trace_id=trace.trace_id,
        trace_ref=trace.trace_ref,
        status="classified",
        flags=flags,
        model=data.model,
        classified_at=existing.classified_at,
    )


async def _process_claimed(
    classifier: SignalClassifier,
    repository: SignalRepositoryProtocol,
    scope: Scope,
    execution: Execution,
    config: SignalConfig,
    claimed_until: datetime,
) -> None:
    from litellm._logging import verbose_proxy_logger

    attempt: Final = await classifier.classify(scope, execution, config)
    try:
        await repository.store(execution, config, claimed_until, classifier.clock(), attempt)
    except Exception as error:
        verbose_proxy_logger.error("Lens signal result could not be stored: %s", redact_internal_details(str(error)))


class _SignalScan:
    def __init__(
        self,
        reader: SourceReader,
        repository: SignalRepositoryProtocol,
        scope: Scope,
        config: SignalConfig,
        now: datetime,
        cursor: str,
        limit: int,
        sweep: SignalSweep,
    ) -> None:
        self.reader: Final = reader
        self.repository: Final = repository
        self.scope: Final = scope
        self.config: Final = config
        self.now: Final = now
        self.cursor: str = cursor
        self.limit: Final = limit
        self.sweep: Final = sweep
        self.executions: tuple[Execution, ...] = ()
        self.finished: bool = False

    async def _read_page(self, start: int, end: int) -> tuple[tuple[Execution, ...], str | None]:
        page_cursor: Final = self.cursor
        sample: Final = await self.reader.sample(
            self.scope,
            ActivitySelection(source="traces"),
            start,
            end,
            page_size=SIGNAL_PAGE_SIZE,
            cursor=page_cursor,
        )
        identities: Final = tuple(
            TraceIdentity(trace_id=trace.trace_id, trace_ref=trace.trace_ref) for trace in sample.executions
        )
        existing_rows: Final = await self.repository.traces(identities)
        existing: Final = MappingProxyType({signal_identity(row): row for row in existing_rows})
        remaining: Final = self.limit - len(self.executions)
        all_eligible: Final = tuple(
            execution
            for execution in sample.executions
            if candidate(execution, existing.get(signal_identity(execution)), self.config.key(), self.now)
        )
        eligible: Final = all_eligible[:remaining]
        next_cursor: Final = page_cursor if len(all_eligible) > remaining else sample.next_cursor
        return eligible, next_cursor

    async def run(self) -> tuple[tuple[Execution, ...], str]:
        start: Final = int((self.now - self.sweep.lookback).timestamp() * 1000)
        end: Final = int((self.now - SIGNAL_SETTLE).timestamp() * 1000)
        for _ in range(self.sweep.max_pages):
            if self.finished or len(self.executions) >= self.limit:
                break
            eligible, next_cursor = await self._read_page(start, end)
            self.executions = (*self.executions, *eligible)
            if next_cursor is None:
                self.cursor = ""
                self.finished = True
            else:
                self.cursor = next_cursor
        return self.executions, self.cursor


async def _scan_pages(
    reader: SourceReader,
    repository: SignalRepositoryProtocol,
    scope: Scope,
    config: SignalConfig,
    now: datetime,
    cursor: str,
    remaining: int,
    sweep: SignalSweep,
) -> tuple[tuple[Execution, ...], str]:
    if remaining <= 0:
        return (), cursor
    scan: Final = _SignalScan(reader, repository, scope, config, now, cursor, remaining, sweep)
    return await scan.run()


@dataclass(frozen=True, slots=True)
class SignalTick:
    cursor: str
    claimed: int = 0


async def run_signal_tick(
    storage: Storage,
    repository: SignalRepositoryProtocol | None,
    completion: DecisionsCall | None,
    clock: Clock,
    router_ready: RouterReady = lambda: True,
    cursor: str = "",
    sweep: SignalSweep = SIGNAL_BACKLOG_SWEEP,
) -> SignalTick:
    if repository is None or completion is None or not router_ready():
        return SignalTick(cursor)
    now: Final = clock()
    config: Final = await repository.get_config()
    if not config.enabled:
        return SignalTick(cursor)
    reader: Final = SourceReader(storage)
    scope: Final = Scope(all_teams=True)
    candidates: Final = await _scan_pages(
        reader,
        repository,
        scope,
        config,
        now,
        cursor,
        SIGNAL_MAX_PER_TICK,
        sweep,
    )
    executions, next_cursor = candidates
    classifier: Final = SignalClassifier(reader, completion, clock)
    semaphore: Final = asyncio.Semaphore(SIGNAL_CONCURRENCY)

    async def process(execution: Execution) -> bool:
        from litellm._logging import verbose_proxy_logger

        async with semaphore:
            claimed_at: Final = classifier.clock()
            claimed_until: Final = claimed_at + SIGNAL_CLAIM_LEASE
            try:
                claimed: Final = await repository.claim(execution, config, claimed_until, claimed_at)
            except Exception as error:
                verbose_proxy_logger.error("Lens signal claim failed: %s", redact_internal_details(str(error)))
                return False
            if not claimed:
                return False
            await _process_claimed(classifier, repository, scope, execution, config, claimed_until)
            return True

    outcomes: Final = await asyncio.gather(*(process(execution) for execution in executions))
    return SignalTick(next_cursor, sum(outcomes))


async def _logged_tick(
    storage: Storage,
    repository: SignalRepositoryProtocol | None,
    completion: DecisionsCall | None,
    clock: Clock,
    router_ready: RouterReady,
    cursor: str,
    sweep: SignalSweep,
) -> SignalTick:
    from litellm._logging import verbose_proxy_logger

    try:
        return await run_signal_tick(storage, repository, completion, clock, router_ready, cursor, sweep)
    except Exception as error:
        verbose_proxy_logger.error("Lens signal tick failed: %s", redact_internal_details(str(error)))
        return SignalTick(cursor)


class _SignalLoopState:
    def __init__(self) -> None:
        self.tick: SignalTick = SignalTick("")


async def run_signal_loop(
    storage: Storage,
    repository: SignalRepositoryProtocol | None,
    completion: DecisionsCall | None,
    clock: Clock = lambda: datetime.now(timezone.utc),
    router_ready: RouterReady = lambda: True,
    sweep: SignalSweep = SIGNAL_BACKLOG_SWEEP,
) -> None:
    state: Final = _SignalLoopState()
    while True:
        state.tick = await _logged_tick(storage, repository, completion, clock, router_ready, state.tick.cursor, sweep)
        await asyncio.sleep(0 if state.tick.claimed >= SIGNAL_MAX_PER_TICK else sweep.interval_seconds)
