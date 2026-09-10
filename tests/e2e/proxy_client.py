"""ProxyClient: the shared proxy operations, DI'd into every client (composition).

A frozen-slots dataclass holding a Transport plus poll config. Clients hold a
ProxyClient and add their own route methods; the lifecycle ResourceManager uses the
ProxyClient's key/customer methods for cleanup. Read-backs are eventually consistent
(proxy_batch_write_at ~60s) so they poll to a deadline.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from datetime import datetime
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel

from e2e_http import (
    AnthropicHeaders,
    AuthHeaders,
    NoBody,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
    UnknownApiError,
    is_ok,
    unwrap,
)
from models import (
    AnthropicMessagesBody,
    AnthropicMessagesResponse,
    ChatBody,
    ChatResponse,
    CostMap,
    CostMapEntry,
    CountTokensBody,
    CountTokensResponse,
    CredentialCreateBody,
    CredentialCreateResponse,
    CustomerDeleteBody,
    EmbedBody,
    EmbedResponse,
    FileListResponse,
    FineTuningJobsParams,
    FineTuningJobsResponse,
    KeyDeleteBody,
    KeyGenerateBody,
    KeyGenerateResponse,
    KeyInfo,
    KeyInfoParams,
    KeyInfoResponse,
    LiteLLMParamsBody,
    ModelDeleteBody,
    ModelInfoBody,
    ModelInfoEntry,
    ModelInfoResponse,
    ModelMode,
    ModelNewBody,
    ModelNewResponse,
    ModelsListParams,
    ModelsListResponse,
    ModelUpdateBody,
    OcrBody,
    OcrResponse,
    SpendLogRow,
    SpendLogs,
    SpendLogsPage,
    SpendLogsPageParams,
    SpendLogsParams,
    ToolsetCreateBody,
    ToolsetRow,
    ToolsetUpdateBody,
)
from e2e_config import (
    CONTROL_PLANE_BASE_URL,
    MASTER_KEY,
    POLL_INTERVAL,
    POLL_TIMEOUT,
    PROXY_BASE_URL,
    PROXY_REPLICA_URLS,
    REQUEST_TIMEOUT,
    SLOW_PROVIDER_TIMEOUT_SECONDS,
    settle_propagation,
)
from transport import HttpTransport, SplitTransport, Transport, is_control_plane_path

RowsPredicate = Callable[[list[SpendLogRow]], bool]

# After /model/new, poll /v1/models on every replica in PROXY_REPLICA_URLS until each
# lists the model (or fail). Bound by MODEL_SERVABLE_TIMEOUT per replica so a stuck
# reload does not burn the spend poll_timeout (120s). Return on first listing;
# settle_propagation owns the separate wait that lets the workers behind each replica
# reload before the caller uses the model.
MODEL_SERVABLE_TIMEOUT = 40.0
MODEL_SERVABLE_DB_SYNC_SECONDS = 0.0
MODEL_SERVABLE_INTERVAL = 2.0
# Cap each /v1/models poll so one slow request cannot outlast the remaining budget.
MODEL_SERVABLE_REQUEST_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class Servable:
    """The data plane listed the model within the deadline."""


@dataclass(frozen=True, slots=True)
class NotServable:
    """The deadline passed without the data plane listing the model.

    `last_result` is the final /v1/models read, so the caller can tell "the proxy
    answered but omitted the model" (propagation) from "the read itself failed"
    (network/auth) when reporting."""

    last_result: Result[ModelsListResponse] | None


ServableOutcome = Servable | NotServable

type ModelsPoller = Callable[[float], Result[ModelsListResponse]]


@dataclass(frozen=True, slots=True)
class NotServableOn:
    """`NotServable` labeled with the replica whose /v1/models never listed the model."""

    replica: str
    last_result: Result[ModelsListResponse] | None


def await_servable(
    list_models: Callable[[float], Result[ModelsListResponse]],
    *,
    model_name: str,
    timeout: float,
    interval: float,
    request_timeout: float,
    db_sync_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> ServableOutcome:
    """Poll until `model_name` is listed long enough for every worker to DB-sync.

    First listing must happen within `timeout`. After that, the model must stay
    listed continuously for `db_sync_seconds` (any miss resets the continuous
    window). `db_sync_seconds=0` returns on the first listing. Each poll's request
    timeout is clamped to the remaining budget. Sleeps only min(interval, time left)
    so a final deadline-clamped poll is never skipped just because a full interval
    does not fit. Clock and sleep are injected."""
    started = now()
    first_seen_at: float | None = None
    last_result: Result[ModelsListResponse] | None = None
    while True:
        t = now()
        phase_deadline = started + timeout if first_seen_at is None else first_seen_at + db_sync_seconds
        remaining = phase_deadline - t
        if remaining <= 0:
            if (
                last_result is not None
                and first_seen_at is not None
                and (db_sync_seconds <= 0 or t - first_seen_at >= db_sync_seconds)
            ):
                return Servable()
            return NotServable(last_result=last_result)

        poll_timeout = min(request_timeout, remaining)
        last_result = list_models(poll_timeout)
        listed = isinstance(last_result, Success) and any(entry.id == model_name for entry in last_result.data.data)
        t = now()
        if not listed:
            first_seen_at = None
        elif first_seen_at is None:
            if t > started + timeout:
                return NotServable(last_result=last_result)
            first_seen_at = t
            if db_sync_seconds <= 0:
                return Servable()
        elif t - first_seen_at >= db_sync_seconds:
            return Servable()

        phase_deadline = started + timeout if first_seen_at is None else first_seen_at + db_sync_seconds
        wait = min(interval, phase_deadline - now())
        if wait > 0:
            sleep(wait)


def await_servable_everywhere(
    pollers: Mapping[str, ModelsPoller],
    *,
    model_name: str,
    timeout: float,
    interval: float,
    request_timeout: float,
    db_sync_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> Servable | NotServableOn:
    """`await_servable` against every replica in turn, each with the full budget, so
    the model is only servable once every replica has listed it."""
    for replica, list_models in pollers.items():
        match await_servable(
            list_models,
            model_name=model_name,
            timeout=timeout,
            interval=interval,
            request_timeout=request_timeout,
            db_sync_seconds=db_sync_seconds,
            now=now,
            sleep=sleep,
        ):
            case NotServable(last_result=last_result):
                return NotServableOn(replica=replica, last_result=last_result)
            case Servable():
                continue
    return Servable()


def servable_timeout_message(
    *,
    model_name: str,
    replica: str,
    timeout: float,
    db_sync_seconds: float,
    last_result: Result[ModelsListResponse] | None,
) -> str:
    last_error = (
        f"; last /v1/models poll did not succeed: {last_result}"
        if last_result is not None and not isinstance(last_result, Success)
        else ""
    )
    return (
        f"model {model_name!r} was created but never became servable on {replica} "
        f"within {timeout}s of first listing (plus {db_sync_seconds}s continuous "
        f"DB sync) after /model/new (control/data-plane propagation or "
        f"STORE_MODEL_IN_DB reload issue){last_error}"
    )


type ReplicaRead[T] = Callable[[float], T]


@dataclass(frozen=True, slots=True)
class EverywhereConverged[T]:
    """Every replica answered with something `settled` accepts, keyed by replica."""

    answers: Mapping[str, T]


@dataclass(frozen=True, slots=True)
class NeverConvergedOn[T]:
    """`replica` ran out its budget without an answer `settled` accepts; `last` is
    its final answer, so the failure can say what that replica still serves."""

    replica: str
    last: T


def _last_answer[T](
    read: ReplicaRead[T],
    *,
    settled: Callable[[T], bool],
    timeout: float,
    interval: float,
    request_timeout: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> T:
    """Poll `read` until `settled` accepts its answer or `timeout` runs out, and
    return the last answer either way. Each read's request timeout is clamped to
    the budget left, and the final poll runs even when less than an interval
    remains, so a deadline never skips the read that would have settled."""
    deadline: Final = now() + timeout
    answer = read(min(request_timeout, timeout))
    while not settled(answer):
        remaining = deadline - now()
        if remaining <= 0:
            return answer
        sleep(min(interval, remaining))
        answer = read(min(request_timeout, remaining))
    return answer


def await_everywhere[T](
    reads: Mapping[str, ReplicaRead[T]],
    *,
    settled: Callable[[T], bool],
    timeout: float,
    interval: float,
    request_timeout: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> EverywhereConverged[T] | NeverConvergedOn[T]:
    """`_last_answer` against every replica in turn, each with the full budget, so a
    write counts as visible only once the last replica reflects it, and stop at the
    first replica that never converges. Clock and sleep are injected."""
    def read_replica(
        outcome: EverywhereConverged[T] | NeverConvergedOn[T],
        item: tuple[str, ReplicaRead[T]],
    ) -> EverywhereConverged[T] | NeverConvergedOn[T]:
        if isinstance(outcome, NeverConvergedOn):
            return outcome
        replica, read = item
        answer: Final = _last_answer(
            read,
            settled=settled,
            timeout=timeout,
            interval=interval,
            request_timeout=request_timeout,
            now=now,
            sleep=sleep,
        )
        if not settled(answer):
            return NeverConvergedOn(replica=replica, last=answer)
        return EverywhereConverged(answers=MappingProxyType({**outcome.answers, replica: answer}))

    initial: Final[EverywhereConverged[T] | NeverConvergedOn[T]] = EverywhereConverged(answers=MappingProxyType({}))
    return reduce(read_replica, reads.items(), initial)


def _is_not_found[R: BaseModel](result: Result[R]) -> bool:
    return isinstance(result, UnknownApiError) and result.status_code == 404


def _status_of[R: BaseModel](result: Result[R]) -> int:
    match result:
        case Success(status_code=status_code) | UnknownApiError(status_code=status_code):
            return status_code
        case _:
            return -1


type Poller[T] = Callable[[], T]


@dataclass(frozen=True, slots=True)
class Converged[T]:
    result: T


@dataclass(frozen=True, slots=True)
class NotConverged[T]:
    """The deadline passed without a read satisfying the predicate; `last_result` is
    the final read, so the caller can tell a stale body from a failed request."""

    last_result: T


type ConvergeOutcome[T] = Converged[T] | NotConverged[T]


def await_converged[T](
    poll: Poller[T],
    *,
    converged: Callable[[T], bool],
    timeout: float,
    interval: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> ConvergeOutcome[T]:
    """Poll until a read satisfies `converged` or `timeout` elapses.

    Polls before testing the deadline, so a zero or already-spent budget still gets one
    attempt, and sleeps only min(interval, time left), so the attempt that lands exactly
    on the deadline is taken rather than skipped. Clock and sleep are injected."""
    deadline: Final = now() + timeout
    while True:
        result = poll()
        if converged(result):
            return Converged(result=result)
        remaining = deadline - now()
        if remaining <= 0:
            return NotConverged(last_result=result)
        sleep(min(interval, remaining))


def await_converged_everywhere[T](
    pollers: Mapping[str, Poller[T]],
    *,
    converged: Callable[[T], bool],
    timeout: float,
    interval: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> Mapping[str, ConvergeOutcome[T]]:
    """`await_converged` against every replica in turn, each with the full budget, so a
    replica that lags behind the one a write landed on is polled until it catches up
    rather than failing on its first stale read."""
    return MappingProxyType(
        {
            replica: await_converged(
                poll, converged=converged, timeout=timeout, interval=interval, now=now, sleep=sleep
            )
            for replica, poll in pollers.items()
        }
    )


def first_lagging_replica[T](
    outcomes: Mapping[str, ConvergeOutcome[T]],
) -> tuple[str, NotConverged[T]] | None:
    return next(
        ((replica, outcome) for replica, outcome in outcomes.items() if isinstance(outcome, NotConverged)),
        None,
    )


def converge_timeout_message(*, what: str, replica: str, timeout: float, last_result: object) -> str:
    return (
        f"{what} on {replica} never converged within {timeout}s of the write "
        f"(control/data-plane propagation issue); last read: {last_result}"
    )


@dataclass(frozen=True, slots=True)
class ProxyClient:
    transport: Transport
    replicas: Mapping[str, Transport]
    control_replicas: Mapping[str, Transport]
    poll_timeout: float = 120.0
    poll_interval: float = 5.0
    model_servable_timeout: float = MODEL_SERVABLE_TIMEOUT
    model_servable_db_sync_seconds: float = MODEL_SERVABLE_DB_SYNC_SECONDS
    model_servable_interval: float = MODEL_SERVABLE_INTERVAL
    model_servable_request_timeout: float = MODEL_SERVABLE_REQUEST_TIMEOUT

    # ---- keys / customers (satisfies lifecycle.ResourceClient) ----------

    def generate_key(self, body: KeyGenerateBody) -> str:
        return unwrap(
            self.transport.post(
                "/key/generate",
                headers=self.transport.master,
                json=body,
                response_type=KeyGenerateResponse,
            )
        ).key

    def delete_key(self, key: str) -> None:
        _ = self.transport.post(
            "/key/delete",
            headers=self.transport.master,
            json=KeyDeleteBody(keys=[key]),
            response_type=NoBody,
        )

    def delete_customers(self, user_ids: list[str]) -> None:
        if not user_ids:
            return
        _ = self.transport.post(
            "/customer/delete",
            headers=self.transport.master,
            json=CustomerDeleteBody(user_ids=user_ids),
            response_type=NoBody,
        )

    def key_info(self, key: str) -> KeyInfo:
        return unwrap(
            self.transport.get(
                "/key/info",
                headers=self.transport.master,
                params=KeyInfoParams(key=key),
                response_type=KeyInfoResponse,
            )
        ).info

    def read_back_everywhere[R: BaseModel](
        self,
        path: str,
        *,
        params: BaseModel,
        response_type: type[R],
        converged: Callable[[Result[R]], bool],
    ) -> Mapping[str, Result[R]]:
        """GET `path` under the master key on every replica in PROXY_REPLICA_URLS (the
        data-plane URL alone when the stack exports no per-gateway addresses), polling
        each to poll_timeout until its read satisfies `converged`. Returns that read per
        replica, or fails naming the first replica that never converged and its last
        read. Behind a load balancer the single address proves one replica converged,
        not all of them; only per-gateway addresses make this a fleet-wide proof."""
        outcomes: Final = await_converged_everywhere(
            {
                url: self._body_poller(transport, path, params, response_type)
                for url, transport in self.replicas.items()
            },
            converged=converged,
            timeout=self.poll_timeout,
            interval=self.poll_interval,
            now=time.monotonic,
            sleep=time.sleep,
        )
        lagging: Final = first_lagging_replica(outcomes)
        if lagging is not None:
            replica, outcome = lagging
            raise AssertionError(
                converge_timeout_message(
                    what=f"GET {path}",
                    replica=replica,
                    timeout=self.poll_timeout,
                    last_result=outcome.last_result,
                )
            )
        return MappingProxyType(
            {replica: outcome.result for replica, outcome in outcomes.items() if isinstance(outcome, Converged)}
        )

    @staticmethod
    def _body_poller[R: BaseModel](
        transport: Transport, path: str, params: BaseModel, response_type: type[R]
    ) -> Poller[Result[R]]:
        return lambda: transport.get(path, headers=transport.master, params=params, response_type=response_type)

    def model_info(self) -> list[ModelInfoEntry]:
        """Every configured deployment with the price the proxy resolved for it
        (config override merged over cost-map defaults)."""
        return unwrap(
            self.transport.get(
                "/model/info",
                headers=self.transport.master,
                params=NoBody(),
                response_type=ModelInfoResponse,
            )
        ).data

    def model_cost_map(self) -> dict[str, CostMapEntry]:
        return unwrap(
            self.transport.get(
                "/public/litellm_model_cost_map",
                headers=self.transport.master,
                params=NoBody(),
                response_type=CostMap,
            )
        ).root

    def list_files(self, key: str) -> Result[FileListResponse]:
        return self.transport.get(
            "/v1/files",
            headers=self.transport.bearer(key),
            params=NoBody(),
            response_type=FileListResponse,
        )

    def list_fine_tuning_jobs(self, key: str, params: FineTuningJobsParams) -> Result[FineTuningJobsResponse]:
        return self.transport.get(
            "/v1/fine_tuning/jobs",
            headers=self.transport.bearer(key),
            params=params,
            response_type=FineTuningJobsResponse,
        )

    def create_model(
        self,
        model_name: str,
        litellm_params: LiteLLMParamsBody,
        mode: ModelMode | None = None,
    ) -> str:
        """Register a deployment under `model_name` and return its proxy-assigned
        model_id, once the model is actually servable on the data plane."""
        return self.register_model(
            ModelNewBody(
                model_name=model_name,
                litellm_params=litellm_params,
                model_info=ModelInfoBody(mode=mode),
            )
        )

    def register_model(self, body: ModelNewBody, listed_for: str | None = None) -> str:
        """`create_model` for deployments that carry more than a mode: access groups,
        team scoping, a pinned id. `listed_for` is the virtual key whose /v1/models
        view must list the deployment before it counts as servable, because a
        team-scoped deployment is listed to its own team and to nobody else, master
        key included; leave it unset for a proxy-wide model.

        /model/new is a control-plane route; the data plane (which serves /chat,
        /ocr, ...) only picks the new model up on its next DB reload, so a call
        issued the instant this returns can race the reload and 400 with "Invalid
        model name passed". We poll the data-plane /v1/models until the model
        appears, then settle for the remainder of the propagation budget.

        Both steps are needed. The poll asks every replica in PROXY_REPLICA_URLS
        directly, so behind the stack's load balancer it proves each gateway serves
        the model rather than whichever one the balancer routed the poll to. It still
        cannot see the workers behind a gateway, nor any replica when only the
        balancer address is configured (every request opens a fresh connection, so
        the caller's next request re-rolls), so waiting out PROPAGATION_TIMEOUT is
        what makes the model safe to use anywhere."""
        model_id = unwrap(
            self.transport.post(
                "/model/new",
                headers=self.transport.master,
                json=body,
                response_type=ModelNewResponse,
            )
        ).model_id
        written_at = time.monotonic()
        try:
            self._await_model_servable(body.model_name, listed_for)
        except BaseException:
            self.delete_model(model_id)
            raise
        settle_propagation(written_at)
        return model_id

    def _await_model_servable(self, model_name: str, listed_for: str | None = None) -> None:
        """Block until every replica lists `model_name`, or fail at model_servable_timeout."""
        headers: Final = self.transport.master if listed_for is None else self.transport.bearer(listed_for)
        outcome: Final = await_servable_everywhere(
            {url: self._models_poller(transport, headers) for url, transport in self.replicas.items()},
            model_name=model_name,
            timeout=self.model_servable_timeout,
            interval=self.model_servable_interval,
            request_timeout=self.model_servable_request_timeout,
            db_sync_seconds=self.model_servable_db_sync_seconds,
            now=time.monotonic,
            sleep=time.sleep,
        )
        match outcome:
            case Servable():
                return
            case NotServableOn(replica=replica, last_result=last_result):
                raise AssertionError(
                    servable_timeout_message(
                        model_name=model_name,
                        replica=replica,
                        timeout=self.model_servable_timeout,
                        db_sync_seconds=self.model_servable_db_sync_seconds,
                        last_result=last_result,
                    )
                )

    @staticmethod
    def _models_poller(transport: Transport, headers: AuthHeaders) -> ModelsPoller:
        return lambda poll_timeout: transport.get(
            "/v1/models",
            headers=headers,
            params=ModelsListParams(),
            response_type=ModelsListResponse,
            timeout=poll_timeout,
        )

    def update_model(self, model_id: str, litellm_params: LiteLLMParamsBody) -> None:
        """Merge `litellm_params` over the deployment `model_id`'s stored params via
        POST /model/update. The proxy overlays only the non-null fields and clears
        its model cache, so a later /model/info read reflects the change (eventually,
        after the reload)."""
        unwrap(
            self.transport.post(
                "/model/update",
                headers=self.transport.master,
                json=ModelUpdateBody(
                    litellm_params=litellm_params,
                    model_info=ModelInfoBody(id=model_id),
                ),
                response_type=NoBody,
            )
        )

    def delete_model(self, model_id: str) -> None:
        result = self.transport.post(
            "/model/delete",
            headers=self.transport.master,
            json=ModelDeleteBody(id=model_id),
            response_type=NoBody,
        )
        if not is_ok(result):
            warnings.warn(f"delete_model({model_id!r}) failed: {result}", stacklevel=2)

    # ---- replica read-back ----------------------------------------------

    def replicas_for(self, path: str) -> Mapping[str, Transport]:
        """The replicas that serve `path`: every data-plane replica for an LLM route,
        and for a management route the control-plane replicas, since the data-plane
        replicas trim management routes and answer them 404. A monolith serves both
        from every replica, so a management read-back polls all of them; a split
        deployment exposes one control-plane address (there is one backend process
        behind it on the stack these suites run against), so it polls that. A
        control plane fronting several backends would need its own replica list to
        prove each one converged, the way PROXY_REPLICA_URLS does for the gateways.
        Never empty: a read-back against no replica would assert nothing and pass."""
        replicas: Final = self.control_replicas if is_control_plane_path(path) else self.replicas
        assert replicas, f"no replica is configured to serve {path}, so a read-back there would prove nothing"
        return replicas

    def read_body_back_everywhere[R: BaseModel](
        self, path: str, response_type: type[R], *, settled: Callable[[R], bool]
    ) -> Mapping[str, R]:
        """GET `path` on every replica that serves it, polling each to poll_timeout
        until `settled` accepts its body, and fail naming the first replica that
        never converged. Returns each replica's settled body, keyed by replica, so
        the caller can assert the rest of it."""
        outcome: Final = await_everywhere(
            {url: self._reader(transport, path, response_type) for url, transport in self.replicas_for(path).items()},
            settled=lambda result: isinstance(result, Success) and settled(result.data),
            timeout=self.poll_timeout,
            interval=self.poll_interval,
            request_timeout=REQUEST_TIMEOUT,
            now=time.monotonic,
            sleep=time.sleep,
        )
        match outcome:
            case EverywhereConverged(answers=answers):
                return MappingProxyType({url: unwrap(result) for url, result in answers.items()})
            case NeverConvergedOn(replica=replica, last=last):
                raise AssertionError(
                    f"GET {path} on {replica} never converged within {self.poll_timeout}s of the write; "
                    f"last read: {last}"
                )

    def gone_everywhere(self, path: str) -> Mapping[str, int]:
        """Poll GET `path` on every replica that serves it until each stops serving
        it, and fail naming the first replica that still does at poll_timeout.
        Returns each replica's final status, so the caller asserts the 404 itself."""
        outcome: Final = await_everywhere(
            {url: self._reader(transport, path, NoBody) for url, transport in self.replicas_for(path).items()},
            settled=_is_not_found,
            timeout=self.poll_timeout,
            interval=self.poll_interval,
            request_timeout=REQUEST_TIMEOUT,
            now=time.monotonic,
            sleep=time.sleep,
        )
        match outcome:
            case EverywhereConverged(answers=answers):
                return MappingProxyType({url: _status_of(result) for url, result in answers.items()})
            case NeverConvergedOn(replica=replica, last=last):
                raise AssertionError(
                    f"GET {path} on {replica} still answers {self.poll_timeout}s after the delete; last read: {last}"
                )

    @staticmethod
    def _reader[R: BaseModel](transport: Transport, path: str, response_type: type[R]) -> ReplicaRead[Result[R]]:
        return lambda request_timeout: transport.get(
            path,
            headers=transport.master,
            params=NoBody(),
            response_type=response_type,
            timeout=request_timeout,
        )

    # ---- mcp toolsets ---------------------------------------------------

    def create_toolset(self, body: ToolsetCreateBody) -> ToolsetRow:
        return unwrap(
            self.transport.post(
                "/v1/mcp/toolset",
                headers=self.transport.master,
                json=body,
                response_type=ToolsetRow,
            )
        )

    def update_toolset(self, body: ToolsetUpdateBody) -> ToolsetRow:
        """PUT /v1/mcp/toolset: a partial update where a field left unset keeps its
        stored value and None clears it."""
        return unwrap(
            self.transport.put(
                "/v1/mcp/toolset",
                headers=self.transport.master,
                json=body,
                response_type=ToolsetRow,
            )
        )

    def delete_toolset(self, toolset_id: str) -> Result[NoBody]:
        """DELETE /v1/mcp/toolset/{toolset_id}. Returns the outcome so the act phase
        can unwrap it while a deferred teardown can ignore an already-deleted row."""
        return self.transport.delete(
            f"/v1/mcp/toolset/{toolset_id}",
            headers=self.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def create_credential(self, body: CredentialCreateBody) -> None:
        unwrap(
            self.transport.post(
                "/credentials",
                headers=self.transport.master,
                json=body,
                response_type=CredentialCreateResponse,
            )
        )

    def delete_credential(self, credential_name: str) -> None:
        result = self.transport.delete(
            f"/credentials/{credential_name}",
            headers=self.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )
        if not is_ok(result):
            warnings.warn(f"delete_credential({credential_name!r}) failed: {result}", stacklevel=2)

    # ---- LLM calls ------------------------------------------------------

    def chat(self, key: str, body: ChatBody) -> Result[ChatResponse]:
        return self.transport.post(
            "/chat/completions",
            headers=self.transport.bearer(key),
            json=body,
            response_type=ChatResponse,
        )

    def chat_stream(self, key: str, body: ChatBody) -> StreamingResponse:
        return self.transport.stream("/chat/completions", headers=self.transport.bearer(key), json=body)

    def messages_stream(self, key: str, body: AnthropicMessagesBody) -> StreamingResponse:
        return self.transport.stream("/v1/messages", headers=self.transport.bearer(key), json=body)

    def embed(self, key: str, body: EmbedBody) -> Result[EmbedResponse]:
        return self.transport.post(
            "/embeddings",
            headers=self.transport.bearer(key),
            json=body,
            response_type=EmbedResponse,
        )

    def ocr(self, key: str, body: OcrBody) -> Result[OcrResponse]:
        return self.transport.post(
            "/v1/ocr",
            headers=self.transport.bearer(key),
            json=body,
            response_type=OcrResponse,
            timeout=SLOW_PROVIDER_TIMEOUT_SECONDS,
        )

    def count_tokens(self, key: str, body: CountTokensBody) -> Result[CountTokensResponse]:
        """POST /v1/messages/count_tokens (Anthropic-native). Sends the
        anthropic-version header so the native path accepts it; harmless on the
        other providers the proxy fronts."""
        return self.transport.post(
            "/v1/messages/count_tokens",
            headers=self._anthropic_headers(key),
            json=body,
            response_type=CountTokensResponse,
        )

    def messages(self, key: str, body: AnthropicMessagesBody) -> Result[AnthropicMessagesResponse]:
        """POST /v1/messages (Anthropic-native). The response is either the
        Anthropic-shape passthrough (`content`) or the OpenAI-normalized shape
        (`choices`); AnthropicMessagesResponse models both."""
        return self.transport.post(
            "/v1/messages",
            headers=self._anthropic_headers(key),
            json=body,
            response_type=AnthropicMessagesResponse,
        )

    def _anthropic_headers(self, key: str) -> AnthropicHeaders:
        return AnthropicHeaders(authorization=self.transport.bearer(key).authorization)

    # ---- spend read-back ------------------------------------------------

    def spend_logs(self, params: SpendLogsParams) -> list[SpendLogRow]:
        result = self.transport.get(
            "/spend/logs",
            headers=self.transport.master,
            params=params,
            response_type=SpendLogs,
        )
        match result:
            case Success(data=logs):
                return logs.root
            case _:
                return []

    def spend_logs_window(self, *, start: datetime, end: datetime) -> list[SpendLogRow]:
        def fetch(page: int) -> SpendLogsPage:
            return unwrap(
                self.transport.get(
                    "/spend/logs/v2",
                    headers=self.transport.master,
                    params=SpendLogsPageParams(
                        start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                        page=page,
                        page_size=100,
                    ),
                    response_type=SpendLogsPage,
                )
            )

        first = fetch(1)
        return [
            *first.data,
            *(row for page in range(2, first.total_pages + 1) for row in fetch(page).data),
        ]

    def poll_logs_for_key(
        self, key: str, *, min_rows: int = 1, predicate: RowsPredicate | None = None
    ) -> list[SpendLogRow]:
        return self._poll(lambda: self.spend_logs(SpendLogsParams(api_key=key)), min_rows, predicate)

    def poll_logs_for_request_id(
        self,
        request_id: str,
        *,
        min_rows: int = 1,
        predicate: RowsPredicate | None = None,
    ) -> list[SpendLogRow]:
        return self._poll(
            lambda: self.spend_logs(SpendLogsParams(request_id=request_id)),
            min_rows,
            predicate,
        )

    def _poll(
        self,
        fetch: Callable[[], list[SpendLogRow]],
        min_rows: int,
        predicate: RowsPredicate | None,
    ) -> list[SpendLogRow]:
        deadline = time.monotonic() + self.poll_timeout
        rows: list[SpendLogRow] = []
        while time.monotonic() < deadline:
            rows = fetch()
            if len(rows) >= min_rows and (predicate is None or predicate(rows)):
                return rows
            time.sleep(self.poll_interval)
        return rows

    # ---- route probe ----------------------------------------------------

    def probe(self, path: str, *, params: NoBody) -> ProbeResult:
        return self.transport.probe(path, params=params)


def build_proxy_client(
    *,
    base_url: str = PROXY_BASE_URL,
    master_key: str = MASTER_KEY,
    control_plane_base_url: str = CONTROL_PLANE_BASE_URL,
    replica_urls: tuple[str, ...] = PROXY_REPLICA_URLS,
) -> ProxyClient:
    """The ProxyClient every suite's client is built from: a SplitTransport that routes
    LLM calls to the data plane (PROXY_BASE_URL) and management/admin calls to the
    control plane (CONTROL_PLANE_BASE_URL), with the shared poll budget. The two
    base URLs are the same for a monolithic proxy, so routing is then a no-op.
    ``replica_urls`` (PROXY_REPLICA_URLS) names every data-plane replica the model
    barrier polls directly; it is the data-plane URL itself unless the stack
    exports each gateway's own address. Management read-backs poll those same
    replicas when the two planes share a base URL (a monolith, where every replica
    serves every route) and the control plane alone when they differ (a split
    deployment, where the data-plane replicas do not serve management routes).

    The endpoints are injectable for callers that resolve the proxy some other
    way than ``e2e_config``'s env names (see ``claude_code/_env.py``); they must
    pass all four together, since a caller that overrides only the data plane
    would leave management calls and the replica poll pointed at the env defaults.

    Test-to-proxy traffic always goes over the wire, in every E2E_FIXTURE_MODE:
    record and replay scope to the proxy's provider-bound calls via the
    provider edge (see provider_edge.py), never to this transport."""
    split = SplitTransport(
        data=HttpTransport(
            base_url=base_url,
            master_key=master_key,
            request_timeout=REQUEST_TIMEOUT,
        ),
        control=HttpTransport(
            base_url=control_plane_base_url,
            master_key=master_key,
            request_timeout=REQUEST_TIMEOUT,
        ),
    )
    replicas: Final = MappingProxyType(
        {
            url: HttpTransport(base_url=url, master_key=master_key, request_timeout=REQUEST_TIMEOUT)
            for url in replica_urls
        }
    )
    control_replicas: Final = (
        replicas if control_plane_base_url == base_url else MappingProxyType({control_plane_base_url: split.control})
    )
    return ProxyClient(
        transport=split,
        replicas=replicas,
        control_replicas=control_replicas,
        poll_timeout=POLL_TIMEOUT,
        poll_interval=POLL_INTERVAL,
    )
