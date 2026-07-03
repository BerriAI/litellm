"""Client for the batches e2e suite: file upload/download and the batch
operations (create / retrieve / cancel / list) over the shared Gateway.

`create_batch` returns the raw HTTP outcome (StreamingResponse) so a 403 model
access denial and a provider-native batch body both surface; the test parses
BatchObject from the body. A `provider` arg routes a call to /{provider}/v1/...,
which the provider-fallback scenario needs (its ids are raw, not model-encoded).
The request/response models are co-located here because only this suite uses them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from e2e_gateway import Gateway, build_gateway
from e2e_http import (
    FileUploadForm,
    NoBody,
    Result,
    StreamingResponse,
    UnknownApiError,
)


class FileObject(BaseModel):
    id: str
    object: str | None = None
    purpose: str | None = None
    bytes: int | None = None
    status: str | None = None
    created_at: int | None = None


class BatchObject(BaseModel):
    id: str
    object: str | None = None
    status: str
    endpoint: str | None = None
    input_file_id: str | None = None
    output_file_id: str | None = None
    completion_window: str | None = None
    created_at: int | None = None
    model: str | None = None


class BatchList(BaseModel):
    object: str | None = None
    data: list[BatchObject] = []


class FileDeleteResponse(BaseModel):
    id: str
    object: str | None = None
    deleted: bool


class BatchCreateBody(BaseModel):
    input_file_id: str
    endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"
    model: str | None = None


class ModelQuery(BaseModel):
    model: str | None = None


def is_model_access_denied(resp: StreamingResponse) -> bool:
    """True if the proxy rejected the call because the key may not access the model."""
    return resp.status_code == 403 and "key_model_access_denied" in resp.body


def is_result_access_denied[R: BaseModel](result: Result[R]) -> bool:
    match result:
        case UnknownApiError(status_code=403, body=body):
            return "key_model_access_denied" in body
        case _:
            return False


@dataclass(frozen=True, slots=True)
class BatchClient:
    gateway: Gateway

    def upload_file(
        self,
        *,
        content: bytes,
        form: FileUploadForm,
        key: str,
        model: str | None = None,
        provider: str | None = None,
    ) -> Result[FileObject]:
        return self.gateway.transport.upload(
            _files_path(provider),
            headers=self.gateway.transport.bearer(key),
            form=form,
            filename="batch_input.jsonl",
            content=content,
            params=ModelQuery(model=model),
            response_type=FileObject,
        )

    def create_batch(
        self, *, body: BatchCreateBody, key: str, provider: str | None = None
    ) -> StreamingResponse:
        return self.gateway.transport.send(
            _batches_path(provider),
            headers=self.gateway.transport.bearer(key),
            json=body,
        )

    def retrieve_batch(
        self, batch_id: str, *, key: str, provider: str | None = None
    ) -> Result[BatchObject]:
        return self.gateway.transport.get(
            f"{_batches_path(provider)}/{batch_id}",
            headers=self.gateway.transport.bearer(key),
            params=NoBody(),
            response_type=BatchObject,
        )

    def cancel_batch(
        self, batch_id: str, *, key: str, provider: str | None = None
    ) -> Result[BatchObject]:
        return self.gateway.transport.post(
            f"{_batches_path(provider)}/{batch_id}/cancel",
            headers=self.gateway.transport.bearer(key),
            json=NoBody(),
            response_type=BatchObject,
        )

    def list_batches(
        self, *, key: str, provider: str | None = None
    ) -> Result[BatchList]:
        return self.gateway.transport.get(
            _batches_path(provider),
            headers=self.gateway.transport.bearer(key),
            params=NoBody(),
            response_type=BatchList,
        )

    def delete_file(
        self, file_id: str, *, key: str, provider: str | None = None
    ) -> Result[FileDeleteResponse]:
        return self.gateway.transport.delete(
            f"{_files_path(provider)}/{file_id}",
            headers=self.gateway.transport.bearer(key),
            json=NoBody(),
            response_type=FileDeleteResponse,
        )


def _files_path(provider: str | None) -> str:
    return f"/{provider}/v1/files" if provider else "/v1/files"


def _batches_path(provider: str | None) -> str:
    return f"/{provider}/v1/batches" if provider else "/v1/batches"


def build_client() -> BatchClient:
    return BatchClient(gateway=build_gateway())
