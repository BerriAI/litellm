"""The ORACLE router: one binding per agentic program, learned from verified, delayed feedback.

A program is one agentic task made of many LLM requests that share a prefix. Its first request is bound
to a model by the decision maker and every later request with the same program id reuses that binding.
When the program finishes, the verifier scores it off the critical path, the score and LiteLLM's spend
for the program become one reward, and the decision maker learns from it in arrival order.

Paper: ORACLE, https://arxiv.org/abs/2607.22465
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, NamedTuple

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.core_helpers import (
    get_or_create_metadata_bucket,  # pyright: ignore[reportUnknownVariableType]  # upstream helper returns a bare dict
)
from litellm.router_strategy.adaptive_router.classifier import classify_prompt
from litellm.router_strategy.oracle_router.config import (
    CHOSEN_MODEL_METADATA_KEY,
    FEEDBACK_HISTORY_SIZE,
    PROGRAM_ID_HEADER,
    PROGRAM_ID_METADATA_KEY,
    PROGRAM_SWEEP_THRESHOLD,
    SESSION_ID_FALLBACK_KEYS,
)
from litellm.router_strategy.oracle_router.decision import DecisionMaker, ProgramContext
from litellm.router_strategy.oracle_router.verifier import ProgramOutcome, Verifier, clamp_score
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import OracleRouterConfig, PreRoutingHookResponse, RequestType
from litellm.types.utils import RoutingDecisionCause, StandardLoggingRoutingDecision

_LEARNING_DECISION_MAKERS: Final[frozenset[str]] = frozenset({"thompson"})
_MESSAGES: Final = TypeAdapter(Sequence[AllMessageValues])
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECTS: Final = TypeAdapter(Sequence[object])


def as_str_mapping(value: object) -> Mapping[str, object]:
    """A typed, read-only view of a request-shaped mapping; empty for anything else."""
    if not isinstance(value, Mapping):
        return _EMPTY
    try:
        return _STR_MAPPING.validate_python(value)
    except ValidationError:
        return _EMPTY


class ProgramKey(NamedTuple):
    """A program is private to the API key that started it: the same id under another key is another program."""

    owner: str | None
    program_id: str


@dataclass(frozen=True, slots=True)
class DeferredCompletion:
    """A completion that arrived before any response of the program was observed; the next one applies it."""

    score: float | None
    cost: float | None
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProgramBinding:
    """One row of the lookup table: program id to model, plus what the proxy has seen of the program."""

    program_id: str
    model: str
    context: ProgramContext
    api_key_hash: str | None
    bound_at: float
    requests: int = 0
    cost: float = 0.0
    last_response_text: str = ""
    last_messages: Sequence[AllMessageValues] = ()
    deferred: DeferredCompletion | None = None


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    program_id: str
    model: str
    request_type: str
    score: float
    cost: float
    verified_at: float


def first_user_text(messages: Sequence[Mapping[str, object]] | None) -> str:
    """The program's initial request: the first user message, text parts only."""
    for message in messages or ():
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence):
            parts = tuple(as_str_mapping(part) for part in _OBJECTS.validate_python(content))
            return "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
        return ""
    return ""


def _metadata(request_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """The request's metadata as callers see it, with the proxy-internal bucket layered on top."""
    return {**as_str_mapping(request_kwargs.get("metadata")), **as_str_mapping(request_kwargs.get("litellm_metadata"))}


def _header(request_kwargs: Mapping[str, object], name: str) -> str | None:
    headers: Final = as_str_mapping(request_kwargs.get("headers"))
    matches: Final = tuple(value for key, value in headers.items() if key.lower() == name)
    return str(matches[0]) if matches else None


def resolve_program_id(request_kwargs: Mapping[str, object], program_id_key: str) -> str | None:
    """The program id of a request: ``metadata[program_id_key]``, then the session id spellings, then the header."""
    metadata: Final = _metadata(request_kwargs)
    candidates: Final = (
        metadata.get(program_id_key),
        *(metadata.get(key) for key in SESSION_ID_FALLBACK_KEYS),
        request_kwargs.get("litellm_session_id"),
        _header(request_kwargs, PROGRAM_ID_HEADER),
    )
    return next((str(candidate) for candidate in candidates if candidate), None)


def _transcript(messages: object, fallback: Sequence[AllMessageValues]) -> Sequence[AllMessageValues]:
    if not messages:
        return fallback
    try:
        return tuple(_MESSAGES.validate_python(messages))
    except ValidationError:
        return fallback


def _api_key_hash(request_kwargs: Mapping[str, object]) -> str | None:
    """The key the proxy authenticated, stamped server-side as ``user_api_key_hash``; None outside the proxy."""
    value: Final = _metadata(request_kwargs).get("user_api_key_hash")
    return str(value) if value else None


def _stamp_internal(
    request_kwargs: dict[str, object],  # mutable-ok: pre-routing contract
    key: str,
    value: object | None,
) -> None:
    """Write a router-internal metadata key where the proxy keeps its own state.

    Any copy the caller supplied is dropped from both buckets first, so the post-call hook only ever reads
    what this router wrote; None clears the key.
    """
    for bucket in (request_kwargs.get("metadata"), request_kwargs.get("litellm_metadata")):
        if isinstance(bucket, dict):
            bucket.pop(key, None)  # pyright: ignore[reportUnknownMemberType]  # caller-supplied dict
    if value is not None:
        _, internal = get_or_create_metadata_bucket(request_kwargs)  # pyright: ignore[reportUnknownVariableType]  # bare dict upstream
        internal[key] = value


class OracleRouter:
    """One instance per ``auto_router/oracle_router`` deployment."""

    def __init__(
        self,
        router_name: str,
        config: OracleRouterConfig,
        decision_maker: DecisionMaker,
        verifier: Verifier,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.router_name: Final[str] = router_name
        self.config: Final[OracleRouterConfig] = config
        self.decision_maker: Final[DecisionMaker] = decision_maker
        self.verifier: Final[Verifier] = verifier
        self._clock: Final = clock
        self._cause: Final[RoutingDecisionCause] = (
            "bandit" if config.decision_maker.type in _LEARNING_DECISION_MAKERS else "classifier_plugin"
        )
        self._bindings: Final[dict[ProgramKey, ProgramBinding]] = {}  # mutable-ok: the lookup table
        self._binding_in_flight: Final[dict[ProgramKey, asyncio.Event]] = {}  # mutable-ok: being bound right now
        self._tasks: Final[set[asyncio.Task[float]]] = set()  # mutable-ok: in-flight verifications
        self._history: Final[deque[FeedbackRecord]] = deque(maxlen=FEEDBACK_HISTORY_SIZE)  # mutable-ok: bounded log
        self._update_lock: Final = asyncio.Lock()
        self.programs_bound: int = 0
        self.programs_completed: int = 0
        self.programs_evicted: int = 0
        self.stateless_requests: int = 0
        self.verifications_failed: int = 0

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self.config.available_models)

    def context_for(self, program_id: str, prompt: str) -> ProgramContext:
        return ProgramContext(program_id=program_id, prompt=prompt, request_type=classify_prompt(prompt))

    def binding(self, program_id: str, owner: str | None = None) -> ProgramBinding | None:
        """The program's row as seen by ``owner``, the key that started it (``user_api_key_hash``)."""
        return self._bindings.get(ProgramKey(owner, program_id))

    def bindings_named(self, program_id: str) -> tuple[ProgramBinding, ...]:
        """Every key's program with this id, oldest first: an admin may complete another key's program."""
        matching: Final = (binding for key, binding in self._bindings.items() if key.program_id == program_id)
        return tuple(sorted(matching, key=lambda binding: binding.bound_at))

    def _evict_expired(self, now: float) -> None:
        deadline: Final = now - self.config.program_ttl_seconds
        expired: Final = tuple(key for key, binding in self._bindings.items() if binding.bound_at < deadline)
        for key in expired:
            self._bindings.pop(key, None)
        overflow: Final = len(self._bindings) + 1 - self.config.max_programs  # +1: room for the program being bound
        oldest: Final = tuple(sorted(self._bindings, key=lambda key: self._bindings[key].bound_at))[: max(overflow, 0)]
        for key in oldest:
            self._bindings.pop(key, None)
        self.programs_evicted += len(expired) + len(oldest)

    async def bind(self, program_id: str, prompt: str, api_key_hash: str | None) -> ProgramBinding:
        """Bind a new program, or return its existing row.

        A program is private to the key that started it: the same id under another key starts that key's
        own program instead of riding on, or completing, someone else's. Concurrent first requests of one
        program (a harness that fans out) wait for the decision in flight instead of each asking the
        decision maker, so a program is bound exactly once.
        """
        key: Final = ProgramKey(api_key_hash, program_id)
        in_flight = self._binding_in_flight.get(key)  # rebind-ok: re-read after each wait
        while in_flight is not None:
            await in_flight.wait()
            in_flight = self._binding_in_flight.get(key)
        existing: Final = self._bindings.get(key)
        if existing is not None:
            return existing
        event: Final = asyncio.Event()
        self._binding_in_flight[key] = event
        try:
            return await self._bind(key, prompt)
        finally:
            self._binding_in_flight.pop(key, None)
            event.set()

    async def _bind(self, key: ProgramKey, prompt: str) -> ProgramBinding:
        now: Final = self._clock()
        if len(self._bindings) >= min(PROGRAM_SWEEP_THRESHOLD, self.config.max_programs):
            self._evict_expired(now)
        context: Final = self.context_for(key.program_id, prompt)
        model: Final = await self.decision_maker.select(context)
        binding: Final = ProgramBinding(
            program_id=key.program_id, model=model, context=context, api_key_hash=key.owner, bound_at=now
        )
        self._bindings[key] = binding
        self.programs_bound += 1
        return binding

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: pre-routing contract
        messages: list[dict[str, object]] | None = None,  # mutable-ok: pre-routing contract
        input: str | list[object] | None = None,  # mutable-ok: pre-routing contract
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse | None:
        """Route the request to its program's model, binding the program on its first request."""
        program_id: Final = resolve_program_id(request_kwargs, self.config.program_id_key)
        prompt: Final = first_user_text(messages)
        chosen_model, request_type = await self._decide(program_id, prompt, _api_key_hash(request_kwargs))
        _stamp_internal(request_kwargs, CHOSEN_MODEL_METADATA_KEY, chosen_model)
        _stamp_internal(request_kwargs, PROGRAM_ID_METADATA_KEY, program_id)
        verbose_router_logger.debug(  # nothing request-derived is logged; the decision is on the request's metadata
            "OracleRouter: routed a %s request", "program" if program_id else "stateless"
        )
        return PreRoutingHookResponse(
            model=chosen_model,
            messages=messages,
            routing_decision=StandardLoggingRoutingDecision(
                router_model_name=self.router_name,
                router_type="oracle",
                routed_model=chosen_model,
                cause=self._cause,
                request_type=request_type.value,
            ),
        )

    async def _decide(self, program_id: str | None, prompt: str, api_key_hash: str | None) -> tuple[str, RequestType]:
        if program_id is None:
            self.stateless_requests += 1
            context: Final = self.context_for("", prompt)
            return await self.decision_maker.select(context), context.request_type
        binding: Final = await self.bind(program_id, prompt, api_key_hash)
        return binding.model, binding.context.request_type

    def observe_request(
        self, program_id: str, cost: float, response_text: str, messages: object = None, owner: str | None = None
    ) -> None:
        """Account one finished LLM request of the program: LiteLLM's spend, the latest answer and transcript."""
        key: Final = ProgramKey(owner, program_id)
        binding: Final = self._bindings.get(key)
        if binding is None:
            return
        transcript: Final = _transcript(messages, binding.last_messages)
        self._bindings[key] = replace(
            binding,
            requests=binding.requests + 1,
            cost=binding.cost + max(cost, 0.0),
            last_response_text=response_text or binding.last_response_text,
            last_messages=transcript,
        )
        deferred: Final = binding.deferred
        if deferred is not None:  # the harness reported before this response was observed
            self.complete(program_id, score=deferred.score, cost=deferred.cost, payload=deferred.payload, owner=owner)

    def complete(
        self,
        program_id: str,
        score: float | None = None,
        cost: float | None = None,
        payload: Mapping[str, object] = _EMPTY,
        owner: str | None = None,
    ) -> asyncio.Task[float] | None:
        """Release the program now and verify it in the background; the task resolves to the verified score.

        A program none of whose requests has been observed yet (feedback that raced the last response's
        logging, or a program whose only request failed) keeps its slot and is completed, with these
        arguments, by its next observed response: there is no answer to verify yet, so nothing is learned.
        """
        key: Final = ProgramKey(owner, program_id)
        binding: Final = self._bindings.get(key)
        if binding is None:
            return None
        if binding.requests == 0:
            self._bindings[key] = replace(binding, deferred=DeferredCompletion(score=score, cost=cost, payload=payload))
            return None
        del self._bindings[key]
        outcome: Final = ProgramOutcome(
            program_id=program_id,
            model=binding.model,
            prompt=binding.context.prompt,
            response_text=binding.last_response_text,
            cost=cost if cost is not None else binding.cost,
            messages=binding.last_messages,
            score=score,
            payload=payload,
        )
        self.programs_completed += 1
        task: Final = asyncio.create_task(self._verify_and_learn(binding, outcome))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _verify_and_learn(self, binding: ProgramBinding, outcome: ProgramOutcome) -> float:
        try:
            score: Final = clamp_score(await self.verifier.verify(outcome))
        except Exception:  # noqa: BLE001  # any verifier failure is counted and must not break the feedback loop
            self.verifications_failed += 1
            verbose_router_logger.exception("OracleRouter: a verification failed")
            return float("nan")
        async with self._update_lock:
            self.decision_maker.update(binding.context, binding.model, score)
        self._history.append(
            FeedbackRecord(
                program_id=binding.program_id,
                model=binding.model,
                request_type=binding.context.request_type.value,
                score=score,
                cost=outcome.cost,
                verified_at=self._clock(),
            )
        )
        return score

    async def drain(self) -> None:
        """Wait for every in-flight verification."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    @property
    def pending_verifications(self) -> int:
        return len(self._tasks)

    def _accuracy_by_model(self) -> Mapping[str, float]:
        scores: Final[dict[str, list[float]]] = {}  # mutable-ok: grouping accumulator for the snapshot
        for record in self._history:
            scores.setdefault(record.model, []).append(record.score)
        return {model: round(sum(values) / len(values), 4) for model, values in scores.items()}

    async def get_state_snapshot(self) -> Mapping[str, object]:
        """In-memory view for the introspection endpoint."""
        active: Final[dict[str, int]] = {}  # mutable-ok: counting accumulator for the snapshot
        for binding in self._bindings.values():
            active[binding.model] = active.get(binding.model, 0) + 1
        return {
            "router_name": self.router_name,
            "available_models": list(self.models),
            "decision_maker": self.decision_maker.snapshot(),
            "verifier": type(self.verifier).__name__,
            "active_programs": len(self._bindings),
            "active_per_model": active,
            "programs_bound": self.programs_bound,
            "programs_completed": self.programs_completed,
            "programs_evicted": self.programs_evicted,
            "stateless_requests": self.stateless_requests,
            "pending_verifications": self.pending_verifications,
            "verifications_failed": self.verifications_failed,
            "recent_accuracy": self._accuracy_by_model(),
            "recent_feedback": [
                {
                    "program_id": record.program_id,
                    "model": record.model,
                    "request_type": record.request_type,
                    "score": record.score,
                    "cost": record.cost,
                }
                for record in tuple(self._history)[-20:]
            ],
        }


def request_type_of(prompt: str) -> RequestType:
    return classify_prompt(prompt)
