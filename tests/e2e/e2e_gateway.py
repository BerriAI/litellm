"""Gateway: the shared proxy operations, DI'd into every client (composition).

A frozen-slots dataclass holding a Transport plus poll config. Clients hold a
Gateway and add their own route methods; the lifecycle ResourceManager uses the
Gateway's key/customer methods for cleanup. Read-backs are eventually consistent
(proxy_batch_write_at ~60s) so they poll to a deadline.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from e2e_http import (
    NoBody,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
    is_ok,
    unwrap,
)
from models import (
    ChatBody,
    ChatResponse,
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


@dataclass(frozen=True, slots=True)
class Gateway:
    transport: Transport
    poll_timeout: float = 120.0
    poll_interval: float = 5.0

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
        it is already present on the first poll, so this adds one request."""
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
        not within poll_timeout (a real propagation/config problem, surfaced here
        instead of as a downstream "Invalid model name passed")."""
        deadline = time.monotonic() + self.poll_timeout
        last_result: Result[ModelsListResponse] | None = None
        while time.monotonic() < deadline:
            last_result = self.transport.get(
                "/v1/models",
                headers=self.transport.master,
                params=NoBody(),
                response_type=ModelsListResponse,
            )
            if isinstance(last_result, Success) and any(
                entry.id == model_name for entry in last_result.data.data
            ):
                return
            time.sleep(self.poll_interval)
        last_error = (
            f"; last /v1/models poll did not succeed: {last_result}"
            if last_result is not None and not isinstance(last_result, Success)
            else ""
        )
        raise AssertionError(
            f"model {model_name!r} was created but never became servable on the data "
            f"plane within {self.poll_timeout}s of /model/new (control/data-plane "
            f"propagation or STORE_MODEL_IN_DB reload issue){last_error}"
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


def build_gateway(
    *,
    base_url: str = PROXY_BASE_URL,
    master_key: str = MASTER_KEY,
    control_plane_base_url: str = CONTROL_PLANE_BASE_URL,
) -> Gateway:
    """The Gateway every suite's client is built from: a SplitTransport that routes
    LLM calls to the data plane (PROXY_BASE_URL) and management/admin calls to the
    control plane (CONTROL_PLANE_BASE_URL), with the shared poll budget. The two
    base URLs are the same for a monolithic proxy, so routing is then a no-op.

    The endpoints are injectable for callers that resolve the proxy some other
    way than ``e2e_config``'s env names (see ``claude_code/_env.py``); they must
    pass all three together, since a caller that overrides only the data plane
    would leave management calls pointed at the env default."""
    return Gateway(
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
