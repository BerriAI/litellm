from __future__ import annotations

import base64
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, NonNegativeInt, PositiveInt


class _FixtureModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True)


class ImageUrlDocument(_FixtureModel):
    type: Literal["image_url"]
    image_url: str


class DocumentUrlDocument(_FixtureModel):
    type: Literal["document_url"]
    document_url: str


MistralOcrDocument = Annotated[ImageUrlDocument | DocumentUrlDocument, Field(discriminator="type")]


class JsonSchemaDefinition(_FixtureModel):
    name: str
    schema_value: JsonValue = Field(alias="schema")
    strict: bool | None = None


class AnnotationFormat(_FixtureModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


class MistralOcrParityInput(_FixtureModel):
    model: str
    document: MistralOcrDocument
    pages: list[NonNegativeInt] | None = None
    include_image_base64: bool | None = None
    image_limit: PositiveInt | None = None
    image_min_size: NonNegativeInt | None = None
    bbox_annotation_format: AnnotationFormat | None = None
    document_annotation_format: AnnotationFormat | None = None
    document_annotation_prompt: str | None = None
    extract_header: bool | None = None
    extract_footer: bool | None = None
    table_format: Literal["markdown", "html"] | None = None
    confidence_scores_granularity: Literal["word", "page", "block"] | None = None
    include_blocks: bool | None = None
    id: str | None = None

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
    litellm_input: MistralOcrParityInput
    provider_response: RecordedHttpResponse
