import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

FairnessQueueRejectReason = Literal["queue_full", "queue_deadline_exceeded", "client_disconnected"]


@dataclass(frozen=True, slots=True)
class QueueTicket:
    model: str
    class_name: str
    request_id: str


@dataclass(frozen=True, slots=True)
class ClassQueueState:
    head_request_id: str | None
    head_enqueued_at: float
    depth: int
    pass_value: float
    yielded: bool


_ClassQueue = tuple[tuple[float, str], ...]


@dataclass(slots=True)
class _ModelQueues:
    queues: dict[str, _ClassQueue] = field(default_factory=dict)
    passes: dict[str, float] = field(default_factory=dict)
    yielded: frozenset[str] = frozenset()
    virtual_time: float = 0.0


class InMemoryFairQueueStore:
    def __init__(self) -> None:
        self._models: dict[str, _ModelQueues] = {}  # mutable-ok: live queue registry, rewritten per enqueue/remove

    def _model(self, model: str) -> _ModelQueues:
        return self._models.setdefault(model, _ModelQueues())

    def enqueue(self, ticket: QueueTicket, enqueued_at: float) -> int:
        state: Final = self._model(ticket.model)
        queue: Final = state.queues.get(ticket.class_name, ())
        if not queue:
            state.passes[ticket.class_name] = max(state.passes.get(ticket.class_name, 0.0), state.virtual_time)
        updated: Final = tuple(sorted((*queue, (enqueued_at, ticket.request_id))))
        state.queues[ticket.class_name] = updated
        return len(updated)

    def remove(self, ticket: QueueTicket) -> None:
        state: Final = self._models.get(ticket.model)
        if state is None:
            return
        queue: Final = state.queues.get(ticket.class_name)
        if not queue:
            return
        state.queues[ticket.class_name] = tuple(entry for entry in queue if entry[1] != ticket.request_id)
        if queue[0][1] == ticket.request_id:
            state.yielded = state.yielded - {ticket.class_name}

    def snapshot(self, model: str, class_names: Sequence[str]) -> Mapping[str, ClassQueueState]:
        state: Final = self._model(model)
        return MappingProxyType(
            {
                class_name: ClassQueueState(
                    head_request_id=(queue[0][1] if (queue := state.queues.get(class_name)) else None),
                    head_enqueued_at=(queue[0][0] if queue else math.inf),
                    depth=len(state.queues.get(class_name, ())),
                    pass_value=state.passes.get(class_name, 0.0),
                    yielded=class_name in state.yielded,
                )
                for class_name in class_names
            }
        )

    def advance(self, model: str, class_name: str, amount: float) -> None:
        state: Final = self._model(model)
        new_pass: Final = state.passes.get(class_name, 0.0) + amount
        state.passes[class_name] = new_pass
        state.virtual_time = new_pass
        state.yielded = frozenset()

    def yield_turn(self, model: str, class_name: str) -> None:
        state: Final = self._model(model)
        waiting: Final = frozenset(name for name, queue in state.queues.items() if queue)
        yielded: Final = state.yielded | {class_name}
        state.yielded = frozenset() if waiting <= yielded else yielded

    def depths(self, model: str) -> Mapping[str, int]:
        state: Final = self._models.get(model)
        if state is None:
            return MappingProxyType({})
        return MappingProxyType({class_name: len(queue) for class_name, queue in state.queues.items()})


@dataclass(frozen=True, slots=True)
class QueueAdmitted:
    waited_seconds: float


@dataclass(frozen=True, slots=True)
class QueueRejected:
    reason: FairnessQueueRejectReason
    waited_seconds: float


QueueOutcome = QueueAdmitted | QueueRejected


class FairQueue:
    def __init__(
        self,
        store: InMemoryFairQueueStore | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store if store is not None else InMemoryFairQueueStore()
        self._clock = clock

    def depths(self, model: str) -> Mapping[str, int]:
        return self._store.depths(model)

    async def wait_for_admission(
        self,
        ticket: QueueTicket,
        weights: Mapping[str, float],
        max_wait_seconds: float,
        max_depth: int,
        poll_interval_seconds: float,
        try_admit: Callable[[], Awaitable[bool]],
        is_cancelled: Callable[[], Awaitable[bool]],
    ) -> QueueOutcome:
        started: Final = self._clock()
        depth: Final = self._store.enqueue(ticket, enqueued_at=time.time())
        if depth > max_depth:
            self._store.remove(ticket)
            return QueueRejected(reason="queue_full", waited_seconds=0.0)
        try:
            while True:
                outcome: QueueOutcome | None = await self._poll_once(
                    ticket, weights, started, max_wait_seconds, poll_interval_seconds, try_admit, is_cancelled
                )
                if outcome is not None:
                    return outcome
        finally:
            self._store.remove(ticket)

    async def _poll_once(
        self,
        ticket: QueueTicket,
        weights: Mapping[str, float],
        started: float,
        max_wait_seconds: float,
        poll_interval_seconds: float,
        try_admit: Callable[[], Awaitable[bool]],
        is_cancelled: Callable[[], Awaitable[bool]],
    ) -> QueueOutcome | None:
        if await is_cancelled():
            return QueueRejected(reason="client_disconnected", waited_seconds=self._clock() - started)
        elapsed: Final = self._clock() - started
        if elapsed >= max_wait_seconds:
            return QueueRejected(reason="queue_deadline_exceeded", waited_seconds=elapsed)
        if _has_turn(ticket, self._store.snapshot(ticket.model, tuple(weights))):
            if await try_admit():
                self._store.advance(
                    ticket.model, ticket.class_name, 1.0 / max(weights.get(ticket.class_name, 0.0), 1e-6)
                )
                return QueueAdmitted(waited_seconds=self._clock() - started)
            self._store.yield_turn(ticket.model, ticket.class_name)
        remaining: Final = max_wait_seconds - (self._clock() - started)
        await asyncio.sleep(max(0.0, min(poll_interval_seconds, remaining)))
        return None


def _has_turn(ticket: QueueTicket, snapshot: Mapping[str, ClassQueueState]) -> bool:
    own: Final = snapshot.get(ticket.class_name)
    if own is None or own.head_request_id != ticket.request_id:
        return False
    active: Final = tuple(_rank(name, state) for name, state in snapshot.items() if state.depth > 0)
    eligible: Final = tuple(entry for entry in active if not snapshot[entry[2]].yielded) or active
    return min(eligible) == _rank(ticket.class_name, own)


def _rank(class_name: str, state: ClassQueueState) -> tuple[float, float, str]:
    return (state.pass_value, state.head_enqueued_at, class_name)
