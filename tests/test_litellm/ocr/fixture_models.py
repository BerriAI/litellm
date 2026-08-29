from __future__ import annotations

import base64
import binascii
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from tests.test_litellm._fixture_models import (
    FixtureModel,
    JsonObject,
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    ParityCase,
    SdkInputBase,
)

__all__ = ("JsonSchemaDefinition", "JsonSchemaResponseFormat", "OcrParityCase", "OcrSdkInputBase")

OcrSdkInputBase = SdkInputBase


class MistralImageUrlValue(FixtureModel):
    url: str
    detail: Literal["low", "auto", "high"] | None = None


class MistralImageUrlDocument(FixtureModel):
    type: Literal["image_url"]
    image_url: str | MistralImageUrlValue


class MistralDocumentUrlDocument(FixtureModel):
    type: Literal["document_url"]
    document_url: str
    document_name: str | None = None


MistralDocument = Annotated[
    MistralImageUrlDocument | MistralDocumentUrlDocument,
    Field(discriminator="type"),
]


MistralModel = Literal[
    "mistral/mistral-ocr-2512",
    "mistral/mistral-ocr-4-0",
    "mistral/mistral-ocr-4-1",
    "mistral/mistral-ocr-4",
    "mistral/mistral-ocr-latest",
    "mistral-ocr-2512",
    "mistral-ocr-4-0",
    "mistral-ocr-4-1",
    "mistral-ocr-4",
    "mistral-ocr-latest",
]


class MistralOcrSdkInput(OcrSdkInputBase):
    model: MistralModel
    document: MistralDocument
    custom_llm_provider: Literal["mistral"] | None = None
    pages: str | list[int] | None = None
    include_image_base64: bool | None = None
    image_limit: int | None = None
    image_min_size: int | None = None
    bbox_annotation_format: JsonSchemaResponseFormat | None = None
    document_annotation_format: JsonSchemaResponseFormat | None = None
    document_annotation_prompt: str | None = None
    extract_header: bool = False
    extract_footer: bool = False
    table_format: Literal["markdown", "html"] | None = None
    confidence_scores_granularity: Literal["page", "word", "block"] | None = None
    include_blocks: bool = True
    id: str | None = None

    @model_validator(mode="after")
    def validate_provider_routing(self) -> Self:
        if not self.model.startswith("mistral/") and self.custom_llm_provider != "mistral":
            raise ValueError("unqualified Mistral models require custom_llm_provider='mistral'")
        if self.document_annotation_prompt is not None and self.document_annotation_format is None:
            raise ValueError("document_annotation_prompt requires document_annotation_format")
        return self


def _validate_reducto_source(source: str) -> str:
    if source.startswith("reducto://"):
        return source
    if not source.startswith("data:"):
        raise ValueError("Reducto documents require a reducto:// id or base64 data URI")
    try:
        header, encoded = source.split(",", 1)
    except ValueError as error:
        raise ValueError("invalid Reducto data URI") from error
    if ";base64" not in header:
        raise ValueError("Reducto data URIs must be base64 encoded")
    try:
        base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid Reducto base64 payload") from error
    return source


class ReductoImageUrlDocument(FixtureModel):
    type: Literal["image_url"]
    image_url: str

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, value: str) -> str:
        return _validate_reducto_source(value)


class ReductoDocumentUrlDocument(FixtureModel):
    type: Literal["document_url"]
    document_url: str

    @field_validator("document_url")
    @classmethod
    def validate_document_url(cls, value: str) -> str:
        return _validate_reducto_source(value)


ReductoDocument = Annotated[
    ReductoImageUrlDocument | ReductoDocumentUrlDocument,
    Field(discriminator="type"),
]

ReductoTableOutputFormat = Literal["html", "json", "md", "jsonbbox", "dynamic", "csv"]
ReductoFormattingInclude = Literal[
    "change_tracking",
    "highlight",
    "comments",
    "hyperlinks",
    "signatures",
    "ignore_watermarks",
]
ReductoBlockType = Literal[
    "Header",
    "Footer",
    "Title",
    "Section Header",
    "Page Number",
    "List Item",
    "Figure",
    "Table",
    "Key Value",
    "Text",
    "Comment",
    "Signature",
]


class ReductoFormatting(FixtureModel):
    add_page_markers: bool = False
    table_output_format: ReductoTableOutputFormat = "dynamic"
    merge_tables: bool = False
    include: list[ReductoFormattingInclude] = Field(default_factory=list)

    @field_validator("include")
    @classmethod
    def validate_unique_include(cls, value: list[ReductoFormattingInclude]) -> list[ReductoFormattingInclude]:
        if len(value) != len(set(value)):
            raise ValueError("formatting.include entries must be unique")
        return value


class ReductoChunking(FixtureModel):
    chunk_mode: Literal["variable", "section", "page", "disabled", "block", "page_sections"] = "disabled"
    chunk_size: int | None = None
    chunk_overlap: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_chunking(self) -> Self:
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_size is not None and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class ReductoRetrieval(FixtureModel):
    chunking: ReductoChunking = Field(default_factory=ReductoChunking)
    filter_blocks: list[ReductoBlockType] = Field(default_factory=list)
    embedding_optimized: bool = False

    @field_validator("filter_blocks")
    @classmethod
    def validate_unique_blocks(cls, value: list[ReductoBlockType]) -> list[ReductoBlockType]:
        if len(value) != len(set(value)):
            raise ValueError("retrieval.filter_blocks entries must be unique")
        return value


class ReductoPageRange(FixtureModel):
    start: int | None = Field(default=None, ge=1)
    end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("page range end must be greater than or equal to start")
        return self


class ReductoTenantThrottling(FixtureModel):
    tenant_id: str = Field(min_length=1, max_length=256)
    max_share: float = Field(default=0.5, gt=0, le=1)


class ReductoHybridVpcSettings(FixtureModel):
    environment: str | None = None


ReductoPageSelection = ReductoPageRange | list[ReductoPageRange] | list[int] | list[str]


class ReductoSettings(FixtureModel):
    ocr_system: Literal["standard", "legacy"] = "standard"
    extraction_mode: Literal["ocr", "hybrid"] = "hybrid"
    force_url_result: bool = False
    force_file_extension: str | None = None
    return_ocr_data: bool = False
    return_images: list[Literal["figure", "table", "page"]] = Field(default_factory=list)
    embed_pdf_metadata: bool = False
    embed_pdf_metadata_dpi: int = Field(default=100, ge=50, le=250)
    persist_results: bool = False
    tenant_throttling: ReductoTenantThrottling | None = None
    timeout: float | None = Field(default=None, gt=0)
    page_range: ReductoPageSelection | None = None
    document_password: str | None = None
    hybrid_vpc: ReductoHybridVpcSettings = Field(default_factory=ReductoHybridVpcSettings)

    @field_validator("return_images")
    @classmethod
    def validate_unique_images(
        cls, value: list[Literal["figure", "table", "page"]]
    ) -> list[Literal["figure", "table", "page"]]:
        if len(value) != len(set(value)):
            raise ValueError("settings.return_images entries must be unique")
        return value


class ReductoParseV3SdkInput(OcrSdkInputBase):
    model: Literal["reducto/parse-v3", "parse-v3"]
    document: ReductoDocument
    custom_llm_provider: Literal["reducto"] | None = None
    formatting: ReductoFormatting = Field(default_factory=ReductoFormatting)
    retrieval: ReductoRetrieval = Field(default_factory=ReductoRetrieval)
    settings: ReductoSettings = Field(default_factory=ReductoSettings)

    @model_validator(mode="after")
    def validate_provider_routing(self) -> Self:
        if self.model == "parse-v3" and self.custom_llm_provider != "reducto":
            raise ValueError("unqualified Reducto models require custom_llm_provider='reducto'")
        return self


class ReductoParseLegacySdkInput(OcrSdkInputBase):
    model: Literal["reducto/parse-legacy", "parse-legacy"]
    document: ReductoDocument
    custom_llm_provider: Literal["reducto"] | None = None
    enhance: JsonObject | None = None

    @model_validator(mode="after")
    def validate_provider_routing(self) -> Self:
        if self.model == "parse-legacy" and self.custom_llm_provider != "reducto":
            raise ValueError("unqualified Reducto models require custom_llm_provider='reducto'")
        return self


class OcrParityCase(ParityCase[MistralOcrSdkInput]):
    pass
