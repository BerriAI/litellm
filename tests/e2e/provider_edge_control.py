from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, Literal

from capture_policy import SCENARIO_BYTES, RequestBudget, ScenarioIdentity, ScenarioOutcome, canonical_scenario_id
from capture_session import CaptureResult, CaptureSession
from capture_store import CaptureLeaseStore
from fixture_bundle import BundleRecorder, Interaction, LoadedBundle, RecordedRequest, RecordedResponse
from provider_edge import EdgeBackend, ProviderRequestObservation, RecordEdge, ReplayEdge, ReplaySource
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["begin", "phase", "observe", "count", "status"]
    node: str = ""
    phase: Literal["setup", "call", "teardown"] | None = None
    passed: bool = False
    marker: str = Field(default="", max_length=256)
    observation_id: str = ""


class ControlReply(BaseModel):
    ok: bool
    error: str | None = None
    observation_id: str | None = None
    count: int | None = None
    mode: Literal["record", "replay"] | None = None


@dataclass(slots=True)
class EdgeController:
    identities: Mapping[str, ScenarioIdentity]
    owner: str
    lease_deadline: int
    store: CaptureLeaseStore | None = None
    recorder: BundleRecorder | None = None
    replay_bundle: LoadedBundle | None = None
    outcome_sink: Callable[[tuple[CaptureResult, ...]], None] | None = None
    request_budgets: Mapping[str, RequestBudget] = field(default_factory=lambda: dict[str, RequestBudget]())
    _replay: ReplaySource | None = field(default=None, init=False)
    _active: str | None = field(default=None, init=False)
    _session: CaptureSession | None = field(default=None, init=False)
    _phases: tuple[tuple[str, bool], ...] = field(default=(), init=False)
    _completed: frozenset[str] = field(default_factory=frozenset, init=False)
    _results: tuple[CaptureResult, ...] = field(default=(), init=False)
    _observations: tuple[tuple[str, ProviderRequestObservation], ...] = field(default=(), init=False)
    _in_flight: int = field(default=0, init=False)
    _recorded_bytes: int = field(default=0, init=False)
    _halted: bool = field(default=False, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        if (self.store is None) == (self.replay_bundle is None) or (self.store is not None and self.recorder is None):
            raise ValueError("controller requires exactly one capture store or replay source")
        if not self.identities or any(
            key != canonical_scenario_id(identity.node) for key, identity in self.identities.items()
        ):
            raise ValueError("controller requires canonical enrolled identities")
        if self.replay_bundle is not None:
            self._replay = ReplaySource(self.replay_bundle, test_key=self.test_key)

    def backend(self, upstream_headers: Mapping[str, Mapping[str, str]]) -> EdgeBackend:
        if self._replay is not None:
            return ReplayEdge(self._replay)
        assert self.recorder is not None
        return RecordEdge(
            self.recorder,
            threading.Lock(),
            test_key=self.test_key,
            before_attempt=self.before_attempt,
            response_finished=self.response_finished,
            response_byte_limit=self.remaining_bytes,
            recording_error=self.recording_error,
            upstream_headers=upstream_headers,
            request_error=self.request_error,
        )

    def request_error(self, body: bytes | None) -> str | None:
        budget: Final = self.request_budgets.get(self.test_key())
        return budget.error(body) if budget is not None else None

    def remaining_bytes(self) -> int:
        with self._lock:
            return max(0, SCENARIO_BYTES - self._recorded_bytes)

    def recording_error(self, request: RecordedRequest, response: RecordedResponse) -> str | None:
        with self._lock:
            size: Final = len(Interaction(request=request, response=response).model_dump_json(indent=2).encode())
            if self._recorded_bytes + size > SCENARIO_BYTES:
                return "scenario exceeds capture byte limit"
            self._recorded_bytes += size
            return None

    def test_key(self) -> str:
        with self._lock:
            if self._active is None:
                raise RuntimeError("no active scenario")
            return self._active

    def begin_request(self) -> str | None:
        with self._lock:
            if self._active is None or self._halted:
                return "no active trusted scenario"
            self._in_flight += 1
            return None

    def end_request(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def before_attempt(self) -> str | None:
        with self._lock:
            return self._session.before_attempt() if self._session is not None else "no active capture session"

    def response_finished(self, response: RecordedResponse) -> None:
        with self._lock:
            if self._session is not None:
                self._session.response_finished(response)

    def observe(self, body: bytes | None) -> None:
        with self._lock:
            for _, observation in self._observations:
                observation.observe(body)

    @property
    def results(self) -> tuple[CaptureResult, ...]:
        with self._lock:
            return self._results

    def command(self, request: ControlRequest) -> ControlReply:
        with self._lock:
            if request.action == "status":
                return ControlReply(
                    ok=not self._halted,
                    count=len(self._completed),
                    mode="record" if self.store is not None else "replay",
                )
            if request.action == "begin":
                return self._begin(request.node)
            if self._active is None or canonical_scenario_id(request.node) != self._active:
                return ControlReply(ok=False, error="scenario is not active")
            if request.action == "phase":
                return self._phase(request)
            if request.action == "observe":
                if not request.marker:
                    return ControlReply(ok=False, error="observation marker is empty")
                identifier: Final = uuid.uuid4().hex
                self._observations = (*self._observations, (identifier, ProviderRequestObservation(request.marker)))
                return ControlReply(ok=True, observation_id=identifier)
            observation: Final = next((obs for key, obs in self._observations if key == request.observation_id), None)
            return (
                ControlReply(ok=True, count=observation.count)
                if observation is not None
                else ControlReply(ok=False, error="unknown observation")
            )

    def _begin(self, raw_node: str) -> ControlReply:
        node: Final = canonical_scenario_id(raw_node)
        if self._active is not None or self._halted or node in self._completed:
            return ControlReply(ok=False, error="scenario already started or run halted")
        identity: Final = self.identities.get(node)
        if identity is None:
            return ControlReply(ok=False, error="scenario is not enrolled")
        if self.store is not None:
            failure: Final = self.store.acquire(
                scenario_key=identity.key, owner=self.owner, expires_at=self.lease_deadline
            )
            if failure is not None:
                self._halted = True
                return ControlReply(ok=False, error=failure.reason)
            self._session = CaptureSession(identity, self.owner, self.store)
        self._active = node
        self._phases = ()
        self._observations = ()
        self._recorded_bytes = 0
        return ControlReply(ok=True)

    def _phase(self, request: ControlRequest) -> ControlReply:
        if request.phase is None or request.phase in dict(self._phases):
            self._halted = True
            return ControlReply(ok=False, error="invalid or duplicate scenario phase")
        if request.phase != "setup" and "setup" not in dict(self._phases):
            self._halted = True
            return ControlReply(ok=False, error="scenario setup outcome missing")
        self._phases = (*self._phases, (request.phase, request.passed))
        if not request.passed:
            self._halted = True
        if request.phase != "teardown":
            return ControlReply(ok=True)
        outcome: Final = ScenarioOutcome(
            **{phase: dict(self._phases).get(phase, False) for phase in ("setup", "call", "teardown")}
        )
        error: Final = self._finish(outcome)
        self._completed = self._completed | {self.test_key()}
        self._active = None
        self._halted = self._halted or error is not None
        if self.outcome_sink is not None:
            self.outcome_sink(self._results)
        return ControlReply(ok=error is None, error=error)

    def _finish(self, outcome: ScenarioOutcome) -> str | None:
        if self._in_flight:
            return "scenario ended with in-flight provider requests"
        if self._session is not None:
            result: Final = self._session.finish(outcome)
            assert self.store is not None
            release: Final = self.store.release(scenario_key=result.identity.key, owner=self.owner)
            error: Final = (
                result.error
                or (release.reason if release is not None else None)
                or ("scenario lifecycle was rejected" if self._halted else None)
            )
            self._results = (*self._results, replace(result, error=error))
            return error
        assert self._replay is not None
        if not (outcome.setup and outcome.call and outcome.teardown):
            return "trusted scenario setup, call and teardown must all pass"
        if self._halted:
            return "scenario lifecycle was rejected"
        return self._replay.leftover_error(self.test_key())


class _ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        server: Final = self.server
        assert isinstance(server, ControlServer)
        try:
            length: Final = int(self.headers.get("content-length", "0"))
            if self.path != "/control" or not 0 < length <= 4096:
                self.send_error(400)
                return
            request: Final = ControlRequest.model_validate_json(self.rfile.read(length))
            reply: Final = server.controller.command(request)
        except (ValueError, ValidationError):
            self.send_error(400)
            return
        encoded: Final = reply.model_dump_json().encode()
        self.send_response(200 if reply.ok else 409)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, controller: EdgeController, *, port: int = 0) -> None:
        self.controller: Final = controller
        super().__init__(("127.0.0.1", port), _ControlHandler)
