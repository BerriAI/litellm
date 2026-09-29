from __future__ import annotations

import base64
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _RecordedHttpModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HttpHeader(_RecordedHttpModel):
    name: str
    value: str


class RecordedHttpResponse(_RecordedHttpModel):
    kind: Literal["http"]
    status_code: int
    headers: tuple[HttpHeader, ...]
    body_b64: str

    @classmethod
    def from_bytes(
        cls,
        status_code: int,
        headers: tuple[HttpHeader, ...],
        body: bytes,
    ) -> RecordedHttpResponse:
        return cls(
            kind="http",
            status_code=status_code,
            headers=headers,
            body_b64=base64.b64encode(body).decode("ascii"),
        )

    def body_bytes(self) -> bytes:
        return base64.b64decode(self.body_b64, validate=True)


class RecordedStreamChunk(_RecordedHttpModel):
    data_b64: str

    @classmethod
    def from_bytes(cls, data: bytes) -> RecordedStreamChunk:
        return cls(data_b64=base64.b64encode(data).decode("ascii"))

    def data_bytes(self) -> bytes:
        return base64.b64decode(self.data_b64, validate=True)


class RecordedHttpStreamResponse(_RecordedHttpModel):
    kind: Literal["http_stream"]
    status_code: int
    headers: tuple[HttpHeader, ...]
    chunks: tuple[RecordedStreamChunk, ...]


RecordedResponse = Annotated[
    RecordedHttpResponse | RecordedHttpStreamResponse,
    Field(discriminator="kind"),
]
