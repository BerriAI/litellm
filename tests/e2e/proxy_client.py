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
)
from transport import HttpTransport, SplitTransport, Transport

RowsPredicate = Callable[[list[SpendLogRow]], bool]

# After /model/new, poll the data plane until the model is listed (or fail).
# Shorter than poll_timeout (spend/log read-backs ~120s); longer than a single
# request. 40s is the harness middle ground: happy path returns on the first
# poll, a stuck reload fails in under a minute instead of two.
MODEL_SERVABLE_TIMEOUT = 40.0
MODEL_SERVABLE_INTERVAL = 2.0
# Cap each /v1/models poll so one slow request cannot outlast the budget.
# Clamped further to remaining deadline inside await_servable.
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
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> ServableOutcome:
    """Poll `list_models` until the data plane lists `model_name` or `timeout` elapses.

    `list_models` receives the per-poll request timeout, clamped to the remaining
    deadline so a slow final poll cannot overrun the overall budget. Clock and sleep
    are injected so this is exercised without wall-clock waits. Always polls at least
    once when the loop starts with a positive budget."""
    deadline = now() + timeout
    last_result: Result[ModelsListResponse] | None = None
    while True:
        remaining = deadline - now()
        if remaining <= 0 and last_result is not None:
            return NotServable(last_result=last_result)
        poll_timeout = min(request_timeout, remaining) if remaining > 0 else request_timeout
        last_result = list_models(poll_timeout)
        if isinstance(last_result, Success) and any(
            entry.id == model_name for entry in last_result.data.data
        ):
            return Servable()
        if now() + interval >= deadline:
            return NotServable(last_result=last_result)
        sleep(interval)


def servable_timeout_message(
    *,
    model_name: str,
    timeout: float,
    last_result: Result[ModelsListResponse] | None,
) -> str:
    last_error = (
        f"; last /v1/models poll did not succeed: {last_result}"
        if last_result is not None and not isinstance(last_result, Success)
        else ""
    )
    return (
        f"model {model_name!r} was created but never became servable on the data "
        f"plane within {timeout}s of /model/new (control/data-plane propagation or "
        f"STORE_MODEL_IN_DB reload issue){last_error}"
    )


@dataclass(frozen=True, slots=True)
class ProxyClient:
    transport: Transport
    poll_timeout: float = 120.0
    poll_interval: float = 5.0
    model_servable_timeout: float = MODEL_SERVABLE_TIMEOUT
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
        model_id, once the model is actually servable on the data plane.

        /model/new is a control-plane route; in a split control/data-plane
        deployment the gateway (data plane, which serves /chat, /ocr, ...) only
        picks the new model up on its next DB reload, so a call issued the instant
        this returns can race the reload and 400 with "Invalid model name passed".
        We therefore poll the data-plane /v1/models until the model appears before
        handing back, so callers can invoke it immediately. In the monolithic case
        it is already present on the first poll, so this adds one request.

        The wait is bounded by `model_servable_timeout` rather than the much longer
        `poll_timeout` used for batched read-backs, so a stuck reload fails in under
        a minute instead of two. Happy path still returns as soon as /v1/models lists
        the model (usually the first poll)."""
        model_id = unwrap(
            self.transport.post(
                "/model/new",
                headers=self.transport.master,
                json=ModelNewBody(
                    model_name=model_name,
                    litellm_params=litellm_params,
                    model_info=ModelInfoBody(mode=mode),
                ),
                response_type=ModelNewResponse,
            )
        ).model_id
        self._await_model_servable(model_name)
        return model_id

    def _await_model_servable(self, model_name: str) -> None:
        """Block until the data plane lists `model_name`, or fail loudly if it does
        not within model_servable_timeout (a real propagation/config problem,
        surfaced here instead of as a downstream "Invalid model name passed")."""
        outcome = await_servable(
            lambda poll_timeout: self.transport.get(
                "/v1/models",
                headers=self.transport.master,
                params=NoBody(),
                response_type=ModelsListResponse,
                timeout=poll_timeout,
            ),
            model_name=model_name,
            timeout=self.model_servable_timeout,
            interval=self.model_servable_interval,
            request_timeout=self.model_servable_request_timeout,
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
    would leave management calls pointed at the env default."""
    return ProxyClient(
        transport=SplitTransport(
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
        ),
        poll_timeout=POLL_TIMEOUT,
        poll_interval=POLL_INTERVAL,
    )
