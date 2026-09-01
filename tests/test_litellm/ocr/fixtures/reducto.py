from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import Annotated, Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from tests.route_parity.fixture_models import FixtureModel, JsonObject
from tests.route_parity.fixtures.recording import ProviderSpec
from tests.test_litellm.ocr.fixtures.base import OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrRecordingTarget,
    fixture_pdf_data_uri,
    invoke_with_api_key,
    parameter_strategy,
    sampled_list_strategy,
    sampled_scalar_strategy,
)


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
ReductoReturnImage = Literal["figure", "table", "page"]
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
_REDUCTO_FILTER_BLOCK_GROUPS: Final[tuple[tuple[ReductoBlockType, ...], ...]] = (
    (),
    ("Header",),
    ("Header", "Footer", "Page Number"),
    ("Figure", "Table", "Key Value"),
)
_REDUCTO_RETURN_IMAGE_GROUPS: Final[tuple[tuple[ReductoReturnImage, ...], ...]] = (
    (),
    ("figure",),
    ("table",),
    ("page",),
    ("figure", "table", "page"),
)


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
ReductoV3Model = Literal["reducto/parse-v3", "parse-v3"]
ReductoLegacyModel = Literal["reducto/parse-legacy", "parse-legacy"]

REDUCTO_V3_MODELS: Final[tuple[Literal["reducto/parse-v3"], ...]] = ("reducto/parse-v3",)
REDUCTO_LEGACY_MODELS: Final[tuple[Literal["reducto/parse-legacy"], ...]] = ("reducto/parse-legacy",)


class ReductoSettings(FixtureModel):
    ocr_system: Literal["standard", "legacy"] = "standard"
    extraction_mode: Literal["ocr", "hybrid"] = "hybrid"
    force_url_result: bool = False
    force_file_extension: str | None = None
    return_ocr_data: bool = False
    return_images: list[ReductoReturnImage] = Field(default_factory=list)
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
    def validate_unique_images(cls, value: list[ReductoReturnImage]) -> list[ReductoReturnImage]:
        if len(value) != len(set(value)):
            raise ValueError("settings.return_images entries must be unique")
        return value


class ReductoParseV3SdkInput(OcrSdkInputBase):
    boundary: str = Field(default="reducto_v3", pattern=r"^reducto_v3$")
    model: ReductoV3Model
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
    boundary: str = Field(default="reducto_legacy", pattern=r"^reducto_legacy$")
    model: ReductoLegacyModel
    document: ReductoDocument
    custom_llm_provider: Literal["reducto"] | None = None
    enhance: JsonObject | None = None

    @model_validator(mode="after")
    def validate_provider_routing(self) -> Self:
        if self.model == "parse-legacy" and self.custom_llm_provider != "reducto":
            raise ValueError("unqualified Reducto models require custom_llm_provider='reducto'")
        return self


_REDUCTO_API_BASE: Final = "https://platform.reducto.ai"


def _formatting_strategy() -> SearchStrategy[ReductoFormatting]:
    values: Final = st.one_of(
        parameter_strategy(
            "table_output_format",
            sampled_scalar_strategy(("dynamic", "html", "md", "json", "csv", "jsonbbox")),
        ),
        parameter_strategy("add_page_markers", sampled_scalar_strategy((False, True))),
        parameter_strategy("merge_tables", sampled_scalar_strategy((False, True))),
        parameter_strategy(
            "include",
            sampled_list_strategy(
                (
                    (),
                    ("hyperlinks",),
                    ("change_tracking", "highlight", "comments"),
                    ("signatures", "ignore_watermarks"),
                )
            ),
        ),
    )
    return values.map(ReductoFormatting.model_validate)


def _chunking_strategy() -> SearchStrategy[ReductoChunking]:
    return st.one_of(
        st.sampled_from(("disabled", "section", "page", "block", "page_sections")).map(
            lambda mode: ReductoChunking(chunk_mode=mode)
        ),
        st.just(ReductoChunking(chunk_mode="variable")),
        sampled_scalar_strategy((250, 1000, 1500)).map(
            lambda size: ReductoChunking(chunk_mode="variable", chunk_size=size)
        ),
        sampled_scalar_strategy((32, 128)).map(
            lambda overlap: ReductoChunking(chunk_mode="variable", chunk_size=1000, chunk_overlap=overlap)
        ),
    )


def _retrieval_strategy() -> SearchStrategy[ReductoRetrieval]:
    filter_blocks: Final[SearchStrategy[list[ReductoBlockType]]] = sampled_list_strategy(_REDUCTO_FILTER_BLOCK_GROUPS)
    return st.one_of(
        _chunking_strategy().map(lambda chunking: ReductoRetrieval(chunking=chunking)),
        filter_blocks.map(lambda selected_blocks: ReductoRetrieval(filter_blocks=selected_blocks)),
        sampled_scalar_strategy((False, True)).map(
            lambda optimized: ReductoRetrieval(
                chunking=ReductoChunking(chunk_mode="variable"),
                embedding_optimized=optimized,
            )
        ),
    )


def _settings_strategy() -> SearchStrategy[ReductoSettings]:
    return_images: Final[SearchStrategy[list[ReductoReturnImage]]] = sampled_list_strategy(_REDUCTO_RETURN_IMAGE_GROUPS)
    page_ranges: Final = st.one_of(
        st.just(ReductoPageRange(start=1, end=1)),
        st.just(ReductoPageRange(start=1, end=3)),
        sampled_list_strategy(
            (
                (
                    ReductoPageRange(start=1, end=2),
                    ReductoPageRange(start=4, end=5),
                ),
            )
        ),
    )
    return st.one_of(
        st.sampled_from(("standard", "legacy")).map(lambda value: ReductoSettings(ocr_system=value)),
        st.sampled_from(("hybrid", "ocr")).map(lambda value: ReductoSettings(extraction_mode=value)),
        st.just(ReductoSettings(force_url_result=True)),
        st.just(ReductoSettings(return_ocr_data=True)),
        return_images.map(lambda selected_images: ReductoSettings(return_images=selected_images)),
        sampled_scalar_strategy((50, 100, 250)).map(
            lambda dpi: ReductoSettings(embed_pdf_metadata=True, embed_pdf_metadata_dpi=dpi)
        ),
        sampled_scalar_strategy((300.0, 900.0)).map(lambda timeout: ReductoSettings(timeout=timeout)),
        page_ranges.map(lambda page_range: ReductoSettings(page_range=page_range)),
    )


def reducto_v3_input_strategy(
    document: ReductoDocumentUrlDocument | None = None,
) -> SearchStrategy[ReductoParseV3SdkInput]:
    selected_document: Final = document or ReductoDocumentUrlDocument(
        type="document_url", document_url="reducto://fixture-document.pdf"
    )
    return st.one_of(
        st.just(ReductoParseV3SdkInput(model="reducto/parse-v3", document=selected_document)),
        st.just(
            ReductoParseV3SdkInput(
                model="parse-v3",
                custom_llm_provider="reducto",
                document=selected_document,
            )
        ),
        _formatting_strategy().map(
            lambda formatting: ReductoParseV3SdkInput(
                model="reducto/parse-v3",
                document=selected_document,
                formatting=formatting,
            )
        ),
        _retrieval_strategy().map(
            lambda retrieval: ReductoParseV3SdkInput(
                model="reducto/parse-v3",
                document=selected_document,
                retrieval=retrieval,
            )
        ),
        _settings_strategy().map(
            lambda settings: ReductoParseV3SdkInput(
                model="reducto/parse-v3",
                document=selected_document,
                settings=settings,
            )
        ),
    )


def reducto_legacy_input_strategy(
    document: ReductoDocumentUrlDocument | None = None,
) -> SearchStrategy[ReductoParseLegacySdkInput]:
    selected_document: Final = document or ReductoDocumentUrlDocument(
        type="document_url", document_url="reducto://fixture-document.pdf"
    )
    return st.sampled_from(
        (
            ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=selected_document),
            ReductoParseLegacySdkInput(model="parse-legacy", custom_llm_provider="reducto", document=selected_document),
            ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=selected_document, enhance={}),
        )
    )


def reducto_recording_targets(environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("REDUCTO_API_KEY")
    if not api_key:
        return ()
    upstream_base: Final = environ.get("REDUCTO_API_BASE", _REDUCTO_API_BASE).rstrip("/")
    document: Final = ReductoDocumentUrlDocument(type="document_url", document_url=fixture_pdf_data_uri())
    invocation: Final = invoke_with_api_key(client, api_key)
    return (
        OcrRecordingTarget(
            name="reducto-v3",
            provider_spec=ProviderSpec(upstream_base=upstream_base),
            strategy=cast(SearchStrategy[OcrSdkInputBase], reducto_v3_input_strategy(document)),
            invocation=invocation,
        ),
        OcrRecordingTarget(
            name="reducto-legacy",
            provider_spec=ProviderSpec(upstream_base=upstream_base),
            strategy=cast(SearchStrategy[OcrSdkInputBase], reducto_legacy_input_strategy(document)),
            invocation=invocation,
        ),
    )
