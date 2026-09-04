"""ProxyClient: the shared proxy operations, DI'd into every client (composition).

A frozen-slots dataclass holding a Transport plus poll config. Clients hold a
ProxyClient and add their own route methods; the lifecycle ResourceManager uses the
ProxyClient's key/customer methods for cleanup. Read-backs are eventually consistent
(proxy_batch_write_at ~60s) so they poll to a deadline.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from e2e_http import (
    AnthropicHeaders,
    NoBody,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
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
    ModelsListResponse,
    ModelUpdateBody,
    OcrBody,
    OcrResponse,
    SpendLogRow,
    SpendLogs,
    SpendLogsPage,
    SpendLogsPageParams,
    SpendLogsParams,
)
from e2e_config import (
    CONTROL_PLANE_BASE_URL,
    MASTER_KEY,
    POLL_INTERVAL,
    POLL_TIMEOUT,
    PROXY_BASE_URL,
    REQUEST_TIMEOUT,
    SLOW_PROVIDER_TIMEOUT_SECONDS,
    settle_propagation,
)
from transport import HttpTransport, SplitTransport, Transport

RowsPredicate = Callable[[list[SpendLogRow]], bool]

# After /model/new, poll data-plane /v1/models until the model is listed (or fail).
# Bound by MODEL_SERVABLE_TIMEOUT so a stuck reload does not burn the spend
# poll_timeout (120s). Return on first listing; settle_propagation owns the separate
# wait that lets every worker and replica reload before the caller uses the model.
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
        phase_deadline = (
            started + timeout if first_seen_at is None else first_seen_at + db_sync_seconds
        )
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
        listed = isinstance(last_result, Success) and any(
            entry.id == model_name for entry in last_result.data.data
        )
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

        phase_deadline = (
            started + timeout if first_seen_at is None else first_seen_at + db_sync_seconds
        )
        wait = min(interval, phase_deadline - now())
        if wait > 0:
            sleep(wait)


def servable_timeout_message(
    *,
    model_name: str,
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
        f"model {model_name!r} was created but never became servable on the data "
        f"plane within {timeout}s of first listing (plus {db_sync_seconds}s continuous "
        f"DB sync) after /model/new (control/data-plane propagation or "
        f"STORE_MODEL_IN_DB reload issue){last_error}"
    )


@dataclass(frozen=True, slots=True)
class ProxyClient:
    transport: Transport
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

    def list_fine_tuning_jobs(
        self, key: str, params: FineTuningJobsParams
    ) -> Result[FineTuningJobsResponse]:
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

        Both steps are needed, and the second is the one that matters at >1 replica.
        The poll proves *a* replica is serving the model; it cannot prove they all
        are, because every request opens a fresh connection and a load-balanced
        Service routes each one independently -- so the caller's next request
        re-rolls and can land on a replica that has not reloaded yet. Waiting out
        PROPAGATION_TIMEOUT is what makes the model safe to use anywhere."""
        model_id = unwrap(
            self.transport.post(
                "/model/new",
                headers=self.transport.master,
                json=body,
                response_type=ModelNewResponse,
            )
        ).model_id
        written_at = time.monotonic()
        self._await_model_servable(body.model_name, listed_for)
        settle_propagation(written_at)
        return model_id

    def _await_model_servable(self, model_name: str, listed_for: str | None = None) -> None:
        """Block until the data plane lists `model_name`, or fail at model_servable_timeout."""
        headers = self.transport.master if listed_for is None else self.transport.bearer(listed_for)
        outcome = await_servable(
            lambda poll_timeout: self.transport.get(
                "/v1/models",
                headers=headers,
                params=NoBody(),
                response_type=ModelsListResponse,
                timeout=poll_timeout,
            ),
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
            case NotServable(last_result=last_result):
                raise AssertionError(
                    servable_timeout_message(
                        model_name=model_name,
                        timeout=self.model_servable_timeout,
                        db_sync_seconds=self.model_servable_db_sync_seconds,
                        last_result=last_result,
                    )
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
) -> ProxyClient:
    """The ProxyClient every suite's client is built from: a SplitTransport that routes
    LLM calls to the data plane (PROXY_BASE_URL) and management/admin calls to the
    control plane (CONTROL_PLANE_BASE_URL), with the shared poll budget. The two
    base URLs are the same for a monolithic proxy, so routing is then a no-op.

    The endpoints are injectable for callers that resolve the proxy some other
    way than ``e2e_config``'s env names (see ``claude_code/_env.py``); they must
    pass all three together, since a caller that overrides only the data plane
    would leave management calls pointed at the env default.

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
    return ProxyClient(
        transport=split,
        poll_timeout=POLL_TIMEOUT,
        poll_interval=POLL_INTERVAL,
    )
