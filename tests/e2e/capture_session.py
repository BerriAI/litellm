from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Protocol

from capture_policy import ScenarioIdentity, ScenarioOutcome, publication_error
from fixture_bundle import RecordedResponse


@dataclass(frozen=True, slots=True)
class AttemptReserved:
    attempt_id: str


@dataclass(frozen=True, slots=True)
class AttemptDenied:
    reason: str


@dataclass(frozen=True, slots=True)
class AttemptUncertain:
    reason: str


type Reservation = AttemptReserved | AttemptDenied | AttemptUncertain


class AttemptStore(Protocol):
    def reserve(self, *, scenario_key: str, owner: str, attempt_id: str) -> Reservation: ...

    def complete(self, *, scenario_key: str, owner: str, attempt_id: str, successful: bool) -> str | None: ...


@dataclass(frozen=True, slots=True)
class CaptureResult:
    identity: ScenarioIdentity
    owner: str
    attempts: tuple[str, ...]
    outcome: ScenarioOutcome
    response_count: int
    error: str | None

    @property
    def publishable(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class CaptureSession:
    identity: ScenarioIdentity
    owner: str
    store: AttemptStore
    new_attempt_id: Callable[[], str] = lambda: uuid.uuid4().hex
    max_attempts: int = 12
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _attempts: tuple[str, ...] = field(default=(), init=False)
    _responses: tuple[RecordedResponse, ...] = field(default=(), init=False)
    _in_flight: bool = field(default=False, init=False)
    _error: str | None = field(default=None, init=False)
    _finished: CaptureResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.owner or not 1 <= self.max_attempts <= 12:
            raise ValueError("capture needs an owner and an attempt cap between 1 and 12")

    def before_attempt(self) -> str | None:
        with self._lock:
            if self._finished is not None:
                return "scenario is closed"
            if self._error is not None:
                return self._error
            if self._in_flight:
                self._error = "concurrent scenario request rejected"
                return self._error
            if len(self._attempts) >= self.max_attempts:
                self._error = "capture attempt cap reached"
                return self._error
            attempt_id: Final = self.new_attempt_id()
            reservation: Final = self.store.reserve(
                scenario_key=self.identity.key, owner=self.owner, attempt_id=attempt_id
            )
            if not isinstance(reservation, AttemptReserved):
                self._error = reservation.reason
                return self._error
            if reservation.attempt_id != attempt_id or attempt_id in self._attempts:
                self._error = "reservation identity mismatch"
                return self._error
            self._attempts = (*self._attempts, attempt_id)
            self._in_flight = True
            return None

    def response_finished(self, response: RecordedResponse) -> None:
        with self._lock:
            if not self._in_flight or self._finished is not None:
                self._error = "response does not belong to an active attempt"
                return
            self._responses = (*self._responses, response)
            self._error = self._error or publication_error(ScenarioOutcome(True, True, True), self._responses)
            completion_error: Final = self.store.complete(
                scenario_key=self.identity.key,
                owner=self.owner,
                attempt_id=self._attempts[-1],
                successful=self._error is None,
            )
            self._error = self._error or completion_error
            self._in_flight = completion_error is not None

    def finish(self, outcome: ScenarioOutcome) -> CaptureResult:
        with self._lock:
            if self._finished is not None:
                return self._finished
            error: Final = (
                self._error
                or ("outbound outcome is uncertain" if self._in_flight else None)
                or publication_error(outcome, self._responses)
            )
            self._finished = CaptureResult(
                self.identity, self.owner, self._attempts, outcome, len(self._responses), error
            )
            return self._finished
