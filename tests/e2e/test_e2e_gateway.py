"""Unit coverage for the Gateway model-management surface (create_model /
delete_model) and the bounded spend read-back (spend_logs_window).

The batches conftest and several llm_translation tests register deployments at
runtime through gateway.create_model; when that method went missing, every batch
test errored at fixture setup (AttributeError) before a single request reached
the proxy. This pins the surface with a typed fake Transport so a rename or
signature drift fails here instead of in a live stage run.

spend_logs_window exists because the unpaginated /spend/logs whole-table read
grew past the e2e runner's memory limit on stage and OOMKilled every run; these
tests pin its /spend/logs/v2 pagination and that SpendLogsParams can no longer
express the unfiltered read.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from batches.batch_client import BatchClient
from e2e_gateway import Gateway
from e2e_http import (
    AuthHeaders,
    FileUploadForm,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
    UnknownApiError,
)
from models import (
    LiteLLMParamsBody,
    ModelDeleteBody,
    ModelNewBody,
    ModelNewResponse,
    ModelsListResponse,
    SpendLogsPage,
    SpendLogsPageParams,
    SpendLogsParams,
)


@dataclass
class _RecordingTransport:
    """Typed fake fulfilling the Transport protocol; records every post and
    answers with a canned success so the test asserts on what was sent.

    `get("/v1/models")` reports a created model as servable only after
    `servable_after_gets` polls, so a test can drive the data-plane wait in
    create_model."""

    posts: list[tuple[str, BaseModel]] = field(default_factory=list)
    servable_after_gets: int = 0
    models_error: UnknownApiError | None = None
    model_gets: int = 0
    spend_total: int = 0
    spend_gets: list[SpendLogsPageParams] = field(default_factory=list)
    _created: list[str] = field(default_factory=list)

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        self.posts.append((path, json))
        if path == "/model/new" and isinstance(json, ModelNewBody):
            self._created.append(json.model_name)
        payload = (
            {"model_id": "registered-id"} if response_type is ModelNewResponse else {}
        )
        return Success(data=response_type.model_validate(payload))

    def stream(
        self, path: str, *, headers: BaseModel, json: BaseModel
    ) -> StreamingResponse:
        raise AssertionError("stream is not part of model management")

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        raise AssertionError("send is not part of model management")

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
    ) -> Result[R]:
        if path == "/v1/models" and response_type is ModelsListResponse:
            self.model_gets += 1
            if self.models_error is not None:
                return self.models_error
            visible = self._created if self.model_gets > self.servable_after_gets else []
            return Success(
                data=response_type.model_validate({"data": [{"id": name} for name in visible]})
            )
        if path == "/spend/logs/v2" and response_type is SpendLogsPage:
            assert isinstance(params, SpendLogsPageParams)
            self.spend_gets.append(params)
            offset = (params.page - 1) * params.page_size
            count = min(params.page_size, max(self.spend_total - offset, 0))
            return Success(
                data=response_type.model_validate(
                    {
                        "data": [{"request_id": f"req-{offset + i}"} for i in range(count)],
                        "total": self.spend_total,
                        "page": params.page,
                        "page_size": params.page_size,
                        "total_pages": (self.spend_total + params.page_size - 1) // params.page_size,
                    }
                )
            )
        raise AssertionError(f"unexpected get: {path}")

    def delete[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        raise AssertionError("delete is not part of model management")

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        raise AssertionError("probe is not part of model management")

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: FileUploadForm,
        filename: str,
        content: bytes,
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        raise AssertionError("upload is not part of model management")

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        raise AssertionError("download is not part of model management")

    def bearer(self, key: str) -> AuthHeaders:
        return AuthHeaders(authorization=f"Bearer {key}")

    @property
    def master(self) -> AuthHeaders:
        return self.bearer("sk-test-master")


def test_gateway_create_model_registers_deployment_and_returns_model_id() -> None:
    transport = _RecordingTransport()
    gateway = Gateway(transport=transport, poll_interval=0.0)

    model_id = gateway.create_model(
        "e2e-test-model", LiteLLMParamsBody(model="openai/gpt-4o-mini")
    )

    assert model_id == "registered-id"
    path, body = transport.posts[0]
    assert path == "/model/new"
    assert isinstance(body, ModelNewBody)
    assert body.model_name == "e2e-test-model"
    # No pinned model_id: the proxy assigns a unique one, so a fixed-name model
    # re-registered after a failed teardown can't collide on the id constraint.
    assert body.model_info.id is None
    assert body.model_info.mode is None
    # It confirmed data-plane visibility before returning.
    assert transport.model_gets >= 1


def test_gateway_create_model_waits_until_servable_on_the_data_plane() -> None:
    # The model shows up on /v1/models only on the third poll (simulating the
    # gateway's delayed DB reload in a split deployment); create_model must keep
    # polling instead of returning after /model/new.
    transport = _RecordingTransport(servable_after_gets=2)
    gateway = Gateway(transport=transport, poll_interval=0.0)

    gateway.create_model("e2e-late-model", LiteLLMParamsBody(model="openai/gpt-4o-mini"))

    assert transport.model_gets == 3


def test_gateway_create_model_fails_loudly_when_never_servable() -> None:
    transport = _RecordingTransport(servable_after_gets=10**9)
    gateway = Gateway(transport=transport, poll_timeout=0.05, poll_interval=0.0)

    with pytest.raises(AssertionError, match="never became servable"):
        gateway.create_model("e2e-ghost-model", LiteLLMParamsBody(model="openai/gpt-4o-mini"))


def test_gateway_create_model_surfaces_the_last_data_plane_error() -> None:
    transport = _RecordingTransport(
        models_error=UnknownApiError(status_code=503, body="data plane down")
    )
    gateway = Gateway(transport=transport, poll_timeout=0.05, poll_interval=0.0)

    with pytest.raises(AssertionError, match="data plane down") as excinfo:
        gateway.create_model("e2e-flaky-model", LiteLLMParamsBody(model="openai/gpt-4o-mini"))
    assert "503" in str(excinfo.value)


def test_batch_client_create_model_registers_a_batch_mode_deployment() -> None:
    transport = _RecordingTransport()
    client = BatchClient(gateway=Gateway(transport=transport, poll_interval=0.0))

    model_id = client.create_model(
        "e2e-batch-model", LiteLLMParamsBody(model="openai/gpt-4o-mini")
    )

    assert model_id == "registered-id"
    path, body = transport.posts[0]
    assert path == "/model/new"
    assert isinstance(body, ModelNewBody)
    assert body.model_info.mode == "batch"


def test_gateway_delete_model_posts_the_model_id() -> None:
    transport = _RecordingTransport()
    gateway = Gateway(transport=transport)

    gateway.delete_model("registered-id")

    path, body = transport.posts[0]
    assert path == "/model/delete"
    assert isinstance(body, ModelDeleteBody)
    assert body.id == "registered-id"


WINDOW_START = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 7, 14, 14, 0, 0, tzinfo=timezone.utc)


def test_gateway_spend_logs_window_pages_through_every_row_in_the_window() -> None:
    transport = _RecordingTransport(spend_total=250)
    gateway = Gateway(transport=transport)

    rows = gateway.spend_logs_window(start=WINDOW_START, end=WINDOW_END)

    assert len(rows) == 250
    assert len({row.request_id for row in rows}) == 250
    assert [params.page for params in transport.spend_gets] == [1, 2, 3]
    assert all(params.start_date == "2026-07-14 12:00:00" for params in transport.spend_gets)
    assert all(params.end_date == "2026-07-14 14:00:00" for params in transport.spend_gets)


def test_gateway_spend_logs_window_stops_at_an_exact_page_boundary() -> None:
    transport = _RecordingTransport(spend_total=200)
    gateway = Gateway(transport=transport)

    rows = gateway.spend_logs_window(start=WINDOW_START, end=WINDOW_END)

    assert len(rows) == 200
    assert [params.page for params in transport.spend_gets] == [1, 2]


def test_gateway_spend_logs_window_returns_empty_for_an_empty_window() -> None:
    transport = _RecordingTransport(spend_total=0)
    gateway = Gateway(transport=transport)

    rows = gateway.spend_logs_window(start=WINDOW_START, end=WINDOW_END)

    assert rows == []
    assert [params.page for params in transport.spend_gets] == [1]


def test_spend_logs_params_rejects_the_unfiltered_whole_table_read() -> None:
    with pytest.raises(ValidationError, match="spend_logs_window"):
        SpendLogsParams()
