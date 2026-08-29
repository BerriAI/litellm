from __future__ import annotations

import base64
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

JsonObject = dict[str, JsonValue]


class _FixtureModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True)


class ImageUrlDocument(_FixtureModel):
    type: Literal["image_url"]
    image_url: str


class DocumentUrlDocument(_FixtureModel):
    type: Literal["document_url"]
    document_url: str


OcrDocument = Annotated[ImageUrlDocument | DocumentUrlDocument, Field(discriminator="type")]


class LiteLLMOcrInput(_FixtureModel):
    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True, serialize_by_alias=True)

    __pydantic_extra__: JsonObject = Field(  # pyright: ignore[reportIncompatibleVariableOverride]  # Pydantic typed extras
        init=False
    )
    model: str
    document: OcrDocument
    custom_llm_provider: str | None = None

    def as_sdk_kwargs(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="python", exclude_unset=True))

    def canonical_input(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="json", exclude_unset=True))


class HttpHeader(_FixtureModel):
    name: str
    value: str


class RecordedHttpResponse(_FixtureModel):
    kind: Literal["http"] = "http"
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
            status_code=status_code,
            headers=headers,
            body_b64=base64.b64encode(body).decode("ascii"),
        )

    def body_bytes(self) -> bytes:
        return base64.b64decode(self.body_b64, validate=True)


class OcrParityCase(_FixtureModel):
    litellm_input: LiteLLMOcrInput
    provider_response: RecordedHttpResponse
