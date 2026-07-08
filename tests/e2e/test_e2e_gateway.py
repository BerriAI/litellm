"""Unit coverage for the Gateway model-management surface (create_model /
delete_model).

The batches conftest and several llm_translation tests register deployments at
runtime through gateway.create_model; when that method went missing, every batch
test errored at fixture setup (AttributeError) before a single request reached
the proxy. This pins the surface with a typed fake Transport so a rename or
signature drift fails here instead of in a live stage run.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel

from batches.batch_client import BatchClient
from e2e_gateway import Gateway
from e2e_http import (
    AuthHeaders,
    FileUploadForm,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
)
from models import (
    LiteLLMParamsBody,
    ModelDeleteBody,
    ModelNewBody,
    ModelNewResponse,
)


@dataclass
class _RecordingTransport:
    """Typed fake fulfilling the Transport protocol; records every post and
    answers with a canned success so the test asserts on what was sent."""

    posts: list[tuple[str, BaseModel]] = field(default_factory=list)

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        self.posts.append((path, json))
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
        raise AssertionError("get is not part of model management")

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
    gateway = Gateway(transport=transport)

    model_id = gateway.create_model(
        "e2e-test-model", LiteLLMParamsBody(model="openai/gpt-4o-mini")
    )

    assert model_id == "registered-id"
    path, body = transport.posts[0]
    assert path == "/model/new"
    assert isinstance(body, ModelNewBody)
    assert body.model_name == "e2e-test-model"
    assert body.model_info.id == "e2e-test-model"
    assert body.model_info.mode is None


def test_batch_client_create_model_registers_a_batch_mode_deployment() -> None:
    transport = _RecordingTransport()
    client = BatchClient(gateway=Gateway(transport=transport))

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
