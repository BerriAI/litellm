import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, TypeAlias

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

from litellm.integrations.clickhouse.context import lens_analysis
from litellm.litellm_core_utils.initialize_dynamic_callback_params import inherit_message_logging_privacy
from litellm.litellm_core_utils.secret_redaction import redact_internal_details
from litellm.proxy.lens.models import ActivitySelection, Execution, Record, Scope, TraceIdentity
from litellm.proxy.lens.sources import SourceReader, Storage

SIGNAL_INTERVAL_SECONDS: Final = 60
SIGNAL_PAGE_SIZE: Final = 100
SIGNAL_MAX_PER_TICK: Final = 50
SIGNAL_CONCURRENCY: Final = 8
SIGNAL_CLAIM_LEASE: Final = timedelta(minutes=5)
SIGNAL_RECLASSIFY_AFTER: Final = timedelta(minutes=5)
SIGNAL_RETRY_FAILED_AFTER: Final = timedelta(minutes=30)
SIGNAL_MAX_CONTENT_PAGES: Final = 3
SIGNAL_PART_MAX_CHARS: Final = 2000
SIGNAL_TRANSCRIPT_MAX_CHARS: Final = 40000
SIGNAL_TRANSCRIPT_HEAD_CHARS: Final = 15000
SIGNAL_TRANSCRIPT_TAIL_CHARS: Final = 25000
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
                "signals": tuple(
                    {"id": signal.id, "question": signal.question}
                    for signal in self.signals
                ),
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
    if not steps or remaining <= 0:
        return ()
    first: Final = steps[0]
    if len(first.content) <= remaining:
        return (first, *_take_head(steps[1:], remaining - len(first.content)))
    return (first.model_copy(update=MappingProxyType({"content": first.content[:remaining]})),)


def _take_tail(steps: tuple[SignalStep, ...], remaining: int) -> tuple[SignalStep, ...]:
    if not steps or remaining <= 0:
        return ()
    last: Final = steps[-1]
    if len(last.content) <= remaining:
        return (*_take_tail(steps[:-1], remaining - len(last.content)), last)
    return (last.model_copy(update=MappingProxyType({"content": last.content[-remaining:]})),)


def _bounded_steps(steps: tuple[SignalStep, ...]) -> tuple[SignalStep, ...]:
    if sum(len(step.content) for step in steps) <= SIGNAL_TRANSCRIPT_MAX_CHARS:
        return steps
    head: Final = _take_head(steps, SIGNAL_TRANSCRIPT_HEAD_CHARS)
    tail: Final = _take_tail(steps, SIGNAL_TRANSCRIPT_TAIL_CHARS)
    omitted_count: Final = len(steps) - len(head) - len(tail)
    marker: Final = SignalStep(kind="omitted", name="", content=f"{omitted_count} steps omitted")
    return (*head, marker, *tail)


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
        SignalStep(kind=part.kind, name=part.name, content=part.content[:SIGNAL_PART_MAX_CHARS])
        for part in content.parts
    )
    rest: Final = (
        await _content_pages(reader, scope, execution, content.next_cursor, pages_left - 1)
        if content.next_cursor is not None
        else ()
    )
    return (*current, *rest)


async def signal_state(reader: SourceReader, scope: Scope, execution: Execution) -> DecisionState:
    steps: Final = _bounded_steps(
        await _content_pages(reader, scope, execution, "", SIGNAL_MAX_CONTENT_PAGES)
    )
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


async def _claim_candidates(
    repository: SignalRepositoryProtocol,
    executions: tuple[Execution, ...],
    config: SignalConfig,
    now: datetime,
    existing: Mapping[tuple[str, str], StoredTraceSignal],
    limit: int,
) -> tuple[tuple[Execution, datetime], ...]:
    eligible: Final = tuple(
        execution
        for execution in executions
        if candidate(execution, existing.get(signal_identity(execution)), config.key(), now)
    )[:limit]
    claimed_until: Final = now + SIGNAL_CLAIM_LEASE
    results: Final = await asyncio.gather(
        *(repository.claim(execution, config, claimed_until, now) for execution in eligible)
    )
    return tuple(
        (execution, claimed_until)
        for execution, was_claimed in zip(eligible, results, strict=True)
        if was_claimed
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


async def _scan_pages(
    reader: SourceReader,
    repository: SignalRepositoryProtocol,
    scope: Scope,
    config: SignalConfig,
    now: datetime,
    cursor: str,
    remaining: int,
) -> tuple[tuple[Execution, datetime], ...]:
    if remaining <= 0:
        return ()
    start: Final = int((now - timedelta(hours=24)).timestamp() * 1000)
    end: Final = int((now - timedelta(minutes=2)).timestamp() * 1000)
    sample: Final = await reader.sample(
        scope,
        ActivitySelection(source="traces"),
        start,
        end,
        page_size=SIGNAL_PAGE_SIZE,
        cursor=cursor,
    )
    identities: Final = tuple(TraceIdentity(trace_id=trace.trace_id, trace_ref=trace.trace_ref) for trace in sample.executions)
    existing_rows: Final = await repository.traces(identities)
    existing: Final = MappingProxyType({signal_identity(row): row for row in existing_rows})
    claimed: Final = await _claim_candidates(repository, sample.executions, config, now, existing, remaining)
    next_page: Final = (
        await _scan_pages(
            reader,
            repository,
            scope,
            config,
            now,
            sample.next_cursor,
            remaining - len(claimed),
        )
        if sample.next_cursor is not None and len(claimed) < remaining
        else ()
    )
    return (*claimed, *next_page)


async def run_signal_tick(
    storage: Storage,
    repository: SignalRepositoryProtocol | None,
    completion: DecisionsCall | None,
    clock: Clock,
    router_ready: RouterReady = lambda: True,
) -> None:
    if repository is None or completion is None or not router_ready():
        return
    now: Final = clock()
    config: Final = await repository.get_config()
    if not config.enabled:
        return
    reader: Final = SourceReader(storage)
    scope: Final = Scope(all_teams=True)
    claimed: Final = await _scan_pages(
        reader,
        repository,
        scope,
        config,
        now,
        "",
        SIGNAL_MAX_PER_TICK,
    )
    classifier: Final = SignalClassifier(reader, completion, clock)
    semaphore: Final = asyncio.Semaphore(SIGNAL_CONCURRENCY)

    async def process(execution: Execution, claimed_until: datetime) -> None:
        async with semaphore:
            await _process_claimed(classifier, repository, scope, execution, config, claimed_until)

    await asyncio.gather(*(process(execution, claimed_until) for execution, claimed_until in claimed))


async def run_signal_loop(
    storage: Storage,
    repository: SignalRepositoryProtocol | None,
    completion: DecisionsCall | None,
    clock: Clock = lambda: datetime.now(timezone.utc),
    router_ready: RouterReady = lambda: True,
) -> None:
    from litellm._logging import verbose_proxy_logger

    while True:
        try:
            await run_signal_tick(storage, repository, completion, clock, router_ready)
        except Exception as error:
            verbose_proxy_logger.error("Lens signal tick failed: %s", redact_internal_details(str(error)))
        await asyncio.sleep(SIGNAL_INTERVAL_SECONDS)
