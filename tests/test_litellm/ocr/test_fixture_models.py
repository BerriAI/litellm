from __future__ import annotations

import base64
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final, TypeVar, cast
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx
from hypothesis import find, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DataObject, SearchStrategy
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.llms.azure_ai.ocr.document_intelligence.transformation import AzureDocumentIntelligenceOCRConfig
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig
from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRRequestData,
)
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.llms.reducto.ocr.transformation import ReductoParseLegacyConfig, ReductoParseV3Config
from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig
from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig
from tests.route_parity.fixtures.media import structured_pdf_data_uri
from tests.test_litellm.ocr.conftest import ocr_fixture_marks
from tests.test_litellm.ocr.fixtures.azure import (
    AZURE_DOCUMENT_INTELLIGENCE_MODELS,
    AZURE_DOCUMENT_INTELLIGENCE_RECORDING_MODELS,
    AZURE_MISTRAL_MODELS,
    AzureDocumentIntelligenceOcrSdkInput,
    AzureMistralOcrSdkInput,
    azure_document_intelligence_input_strategy,
    azure_mistral_input_strategy,
)
from tests.test_litellm.ocr.fixtures.base import (
    DocumentUrlDocument,
    ImageUrlDocument,
    ImageUrlValue,
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    OcrSdkInputBase,
)
from tests.test_litellm.ocr.fixtures.mistral import MISTRAL_MODELS, MistralOcrSdkInput, mistral_input_strategy
from tests.test_litellm.ocr.fixtures.models import OcrParityCase, OcrSdkInput
from tests.test_litellm.ocr.fixtures.reducto import (
    REDUCTO_LEGACY_MODELS,
    REDUCTO_V3_MODELS,
    ReductoChunking,
    ReductoDocumentUrlDocument,
    ReductoFormatting,
    ReductoImageUrlDocument,
    ReductoPageRange,
    ReductoParseLegacySdkInput,
    ReductoParseV3SdkInput,
    ReductoRetrieval,
    ReductoSettings,
    reducto_legacy_input_strategy,
    reducto_v3_input_strategy,
)
from tests.test_litellm.ocr.fixtures.vertex import (
    VERTEX_DEEPSEEK_MODELS,
    VERTEX_MISTRAL_MODELS,
    VertexDeepSeekOcrSdkInput,
    VertexMistralOcrSdkInput,
    vertex_deepseek_input_strategy,
    vertex_mistral_input_strategy,
)

COMMON_FIELDS: Final = frozenset(
    {"contract", "model", "document", "custom_llm_provider", "vertex_project", "vertex_location"}
)
SUPPORTED_OCR_PROVIDERS: Final = frozenset({"mistral", "azure_ai", "reducto", "vertex_ai"})
ACTIVE_OCR_MODELS: Final = frozenset(
    (
        *MISTRAL_MODELS,
        *AZURE_MISTRAL_MODELS,
        *AZURE_DOCUMENT_INTELLIGENCE_MODELS,
        *VERTEX_MISTRAL_MODELS,
        *VERTEX_DEEPSEEK_MODELS,
        *REDUCTO_V3_MODELS,
        *REDUCTO_LEGACY_MODELS,
    )
)
_MISTRAL_2512_OR_NEWER: Final = frozenset(
    {
        "mistral/mistral-ocr-2512",
        "mistral/mistral-ocr-3",
        "mistral/mistral-ocr-3-0",
        "mistral/mistral-ocr-4",
        "mistral/mistral-ocr-4-0",
        "mistral/mistral-ocr-4-1",
        "mistral/mistral-ocr-latest",
    }
)
_MISTRAL_4_OR_NEWER: Final = frozenset(
    {
        "mistral/mistral-ocr-4",
        "mistral/mistral-ocr-4-0",
        "mistral/mistral-ocr-4-1",
        "mistral/mistral-ocr-latest",
    }
)
_MISTRAL_OPTION_GROUPS: Final = frozenset(
    {
        frozenset[str](),
        *(
            frozenset({field})
            for field in (
                "pages",
                "include_image_base64",
                "image_limit",
                "image_min_size",
                "bbox_annotation_format",
                "document_annotation_format",
                "extract_header",
                "extract_footer",
                "table_format",
                "confidence_scores_granularity",
                "include_blocks",
            )
        ),
        frozenset({"document_annotation_format", "document_annotation_prompt"}),
        frozenset({"include_blocks", "confidence_scores_granularity"}),
    }
)
_MISTRAL_2505_OPTION_GROUPS: Final = frozenset(
    {
        frozenset[str](),
        *(
            frozenset({field})
            for field in (
                "pages",
                "include_image_base64",
                "image_limit",
                "image_min_size",
                "bbox_annotation_format",
                "document_annotation_format",
                "confidence_scores_granularity",
            )
        ),
        frozenset({"document_annotation_format", "document_annotation_prompt"}),
    }
)
_AZURE_MISTRAL_OPTION_GROUPS: Final = _MISTRAL_2505_OPTION_GROUPS - {
    frozenset({"document_annotation_format", "document_annotation_prompt"})
}
_REDUCTO_FORMATTING_INCLUDE_GROUPS: Final = (
    (),
    ("hyperlinks",),
    ("change_tracking", "highlight", "comments"),
    ("signatures", "ignore_watermarks"),
)
_REDUCTO_FILTER_BLOCK_GROUPS: Final = (
    (),
    ("Header",),
    ("Header", "Footer", "Page Number"),
    ("Figure", "Table", "Key Value"),
)
_REDUCTO_RETURN_IMAGE_GROUPS: Final = (
    (),
    ("figure",),
    ("table",),
    ("page",),
    ("figure", "table"),
)
_FIND_SETTINGS: Final = settings(max_examples=2_000, deadline=None, derandomize=True, database=None)
_FixtureInputT = TypeVar("_FixtureInputT")
INLINE_IMAGE_DATA_URI: Final = "data:image/png;base64,dGVzdA=="
_MapOcrParams = Callable[[dict[str, object], dict[str, object], str], dict[str, object]]
_TransformOcrRequest = Callable[
    [str, DocumentType, dict[str, object], dict[str, object]],
    OCRRequestData,
]
_GetCompleteUrl = Callable[[str | None, str, dict[str, object]], str]


def _transform_with_stubbed_download(
    transform_request: _TransformOcrRequest,
    model: str,
    document: DocumentType,
    mapped: dict[str, object],
) -> OCRRequestData:
    source_key: Final = "image_url" if document["type"] == "image_url" else "document_url"
    source: Final = document[source_key]
    if source.startswith("data:"):
        return transform_request(model, document, mapped, {})
    media_type: Final = "image/png" if document["type"] == "image_url" else "application/pdf"
    with respx.mock(assert_all_called=False) as router:
        router.route(method="GET").mock(
            return_value=httpx.Response(200, content=b"\x00", headers={"content-type": media_type})
        )
        return transform_request(model, document, mapped, {})


def _find_fixture(
    strategy: SearchStrategy[_FixtureInputT],
    predicate: Callable[[_FixtureInputT], bool],
) -> _FixtureInputT:
    return find(strategy, predicate, settings=_FIND_SETTINGS)


def _document_transport(document: ImageUrlDocument | DocumentUrlDocument) -> tuple[str, str]:
    if isinstance(document, ImageUrlDocument):
        source: Final = document.image_url.url if isinstance(document.image_url, ImageUrlValue) else document.image_url
        return document.type, "data" if source.startswith("data:") else "remote"
    return document.type, "data" if document.document_url.startswith("data:") else "remote"


def _normalized_azure_pages(pages: object) -> str:
    if isinstance(pages, str):
        return pages.replace(" ", "")
    assert isinstance(pages, list)
    raw_pages: Final = cast(list[object], pages)
    if all(isinstance(page, int) for page in raw_pages):
        integer_pages: Final = cast(list[int], raw_pages)
        return ",".join(str(page + 1) for page in sorted(set(integer_pages)))
    string_pages: Final = cast(list[str], raw_pages)
    return ",".join(page.strip() for page in string_pages)


def test_structured_pdf_exercises_semantic_ocr_features() -> None:
    encoded: Final = structured_pdf_data_uri().partition(",")[2]
    pdf: Final = base64.b64decode(encoded, validate=True)

    assert pdf.startswith(b"%PDF-1.")
    assert b"/Count 5" in pdf
    assert pdf.count(b"/Subtype /Image") == 3
    assert all(
        marker in pdf
        for marker in (
            b"/Width 120",
            b"/Width 320",
            b"/Width 360",
            b"/Subtype /Highlight",
            b"/Subtype /Link",
            b"/Subtype /Text",
            b"/Title (Quarterly Operations Report)",
        )
    )
    assert b"Invoice Number: INV-2048" in pdf
    assert b"Formula: gross margin" in pdf
    assert b"Approved by: Jordan Lee" in pdf


class _ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    mode: str | None = None
    litellm_provider: str | None = None
    deprecation_date: date | None = None


MODEL_REGISTRY: Final = TypeAdapter(dict[str, dict[str, JsonValue]])


def _provider_fields(model: type[OcrSdkInputBase]) -> set[str]:
    return set(model.model_fields) - COMMON_FIELDS


def _supported_params(config: BaseOCRConfig, model: str) -> set[str]:
    get_supported_params: Final = cast(Callable[[str], list[str]], config.get_supported_ocr_params)
    return set(get_supported_params(model))


def _mistral_input(**params: object) -> MistralOcrSdkInput:
    return MistralOcrSdkInput.model_validate(
        {
            "model": "mistral/mistral-ocr-latest",
            "document": {"type": "image_url", "image_url": "https://example.com/image.png"},
            **params,
        }
    )


def _reducto_document() -> ReductoDocumentUrlDocument:
    return ReductoDocumentUrlDocument(
        type="document_url",
        document_url="reducto://fixture-document.pdf",
    )


def test_fixture_catalogs_match_active_registered_ocr_models() -> None:
    registry_path: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
    registry: Final = MODEL_REGISTRY.validate_json(registry_path.read_text(encoding="utf-8"))
    active_registered: Final = frozenset(
        model
        for model, raw_metadata in registry.items()
        if raw_metadata.get("mode") == "ocr" and raw_metadata.get("litellm_provider") in SUPPORTED_OCR_PROVIDERS
        for metadata in (_ModelRegistryEntry.model_validate(raw_metadata),)
        if metadata.deprecation_date is None or metadata.deprecation_date > date.today()
    )

    assert ACTIVE_OCR_MODELS == active_registered


@pytest.mark.parametrize(
    ("fixture_model", "provider_config", "model"),
    (
        (MistralOcrSdkInput, MistralOCRConfig(), "mistral-ocr-latest"),
        (AzureMistralOcrSdkInput, AzureAIOCRConfig(), "mistral-document-ai-2512"),
        (
            AzureDocumentIntelligenceOcrSdkInput,
            AzureDocumentIntelligenceOCRConfig(),
            "doc-intelligence/prebuilt-layout",
        ),
        (VertexMistralOcrSdkInput, VertexAIOCRConfig(), "mistral-ocr-2505"),
        (VertexDeepSeekOcrSdkInput, VertexAIDeepSeekOCRConfig(), "deepseek-ai/deepseek-ocr-maas"),
        (ReductoParseV3SdkInput, ReductoParseV3Config(), "parse-v3"),
        (ReductoParseLegacySdkInput, ReductoParseLegacyConfig(), "parse-legacy"),
    ),
)
def test_fixture_fields_match_provider_config(
    fixture_model: type[OcrSdkInputBase], provider_config: BaseOCRConfig, model: str
) -> None:
    assert _provider_fields(fixture_model) == _supported_params(provider_config, model)


@pytest.mark.parametrize(
    "sdk_input",
    (
        AzureMistralOcrSdkInput(
            model="azure_ai/mistral-document-ai-2512",
            document=ImageUrlDocument(type="image_url", image_url="data:image/png;base64,AA=="),
        ),
        VertexMistralOcrSdkInput(
            document=ImageUrlDocument(type="image_url", image_url="data:image/png;base64,AA=="),
            vertex_project="project-1",
        ),
        AzureDocumentIntelligenceOcrSdkInput(
            model="azure_ai/doc-intelligence/prebuilt-layout",
            document=ImageUrlDocument(type="image_url", image_url="data:image/png;base64,AA=="),
        ),
        VertexDeepSeekOcrSdkInput(
            document=ImageUrlDocument(type="image_url", image_url="data:image/png;base64,AA=="),
            vertex_project="project-1",
        ),
    ),
)
def test_provider_contract_is_explicit_but_not_forwarded(sdk_input: OcrSdkInput) -> None:
    assert sdk_input.canonical_input()["contract"] == sdk_input.contract
    assert "contract" not in sdk_input.as_sdk_kwargs()


@pytest.mark.parametrize("legacy_key", ("boundary", None))
def test_ocr_parity_case_migrates_legacy_contract_metadata(legacy_key: str | None) -> None:
    litellm_input: Final[dict[str, object]] = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
    }
    if legacy_key is not None:
        litellm_input[legacy_key] = "mistral"

    fixture: Final = OcrParityCase.model_validate({"litellm_input": litellm_input, "provider_responses": ()})

    assert fixture.litellm_input.contract == "mistral"


def test_mistral_input_preserves_omission_and_explicit_boolean_values() -> None:
    omitted: Final = _mistral_input().as_sdk_kwargs()
    explicit: Final = _mistral_input(extract_header=False, include_blocks=True).as_sdk_kwargs()

    assert "extract_header" not in omitted
    assert "include_blocks" not in omitted
    assert explicit["extract_header"] is False
    assert explicit["include_blocks"] is True


def test_mistral_input_supports_document_and_page_variants() -> None:
    nested_image: Final = MistralOcrSdkInput(
        model="mistral/mistral-ocr-4-1",
        document=ImageUrlDocument(
            type="image_url",
            image_url=ImageUrlValue(url="https://example.com/image.png", detail="high"),
        ),
        pages="0,2-4",
    )
    named_document: Final = MistralOcrSdkInput(
        model="mistral/mistral-ocr-2512",
        document=DocumentUrlDocument(
            type="document_url",
            document_url="https://example.com/document.pdf",
            document_name="invoice.pdf",
        ),
    )

    assert nested_image.canonical_input()["document"] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.png", "detail": "high"},
    }
    assert nested_image.as_sdk_kwargs()["pages"] == "0,2-4"
    assert named_document.canonical_input()["document"] == {
        "type": "document_url",
        "document_url": "https://example.com/document.pdf",
        "document_name": "invoice.pdf",
    }


def test_mistral_annotation_schema_serializes_provider_alias() -> None:
    annotation: Final = JsonSchemaResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaDefinition(
            name="invoice",
            schema={"type": "object"},
        ),
    )
    sdk_input: Final = _mistral_input(
        document_annotation_format=annotation,
        document_annotation_prompt="Extract invoice fields",
    )

    assert sdk_input.canonical_input()["document_annotation_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "invoice",
            "schema": {"type": "object"},
        },
    }


def test_mistral_annotation_prompt_requires_format() -> None:
    with pytest.raises(ValidationError, match="requires document_annotation_format"):
        _mistral_input(document_annotation_prompt="Extract invoice fields")


@pytest.mark.parametrize("field", ("extract_header", "extract_footer", "include_blocks"))
def test_mistral_nonnullable_booleans_reject_null(field: str) -> None:
    with pytest.raises(ValidationError):
        _mistral_input(**{field: None})


def test_unqualified_models_require_explicit_provider() -> None:
    with pytest.raises(ValidationError, match="custom_llm_provider='mistral'"):
        MistralOcrSdkInput(
            model="mistral-ocr-latest",
            document=ImageUrlDocument(type="image_url", image_url="https://example.com/image.png"),
        )
    with pytest.raises(ValidationError, match="custom_llm_provider='reducto'"):
        ReductoParseV3SdkInput(model="parse-v3", document=_reducto_document())


@pytest.mark.parametrize("model", tuple(model.removeprefix("mistral/") for model in MISTRAL_MODELS))
def test_unqualified_mistral_models_accept_explicit_provider(model: str) -> None:
    sdk_input: Final = MistralOcrSdkInput.model_validate(
        {
            "model": model,
            "custom_llm_provider": "mistral",
            "document": ImageUrlDocument(type="image_url", image_url="https://example.com/image.png"),
        }
    )

    assert sdk_input.model == model


@pytest.mark.parametrize(
    ("model", "model_type"),
    (("parse-v3", ReductoParseV3SdkInput), ("parse-legacy", ReductoParseLegacySdkInput)),
)
def test_unqualified_reducto_models_accept_explicit_provider(
    model: str, model_type: type[ReductoParseV3SdkInput] | type[ReductoParseLegacySdkInput]
) -> None:
    sdk_input: Final = model_type.model_validate(
        {"model": model, "custom_llm_provider": "reducto", "document": _reducto_document()}
    )

    assert sdk_input.model == model


@pytest.mark.parametrize(
    "document",
    (
        {"type": "image_url", "image_url": "data:image/png;base64,AA=="},
        {"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
    ),
)
def test_vertex_deepseek_request_maps_both_document_types_to_image_content(
    document: DocumentType,
) -> None:
    request: Final = VertexAIDeepSeekOCRConfig().transform_ocr_request(  # pyright: ignore[reportUnknownMemberType]
        model="deepseek-ai/deepseek-ocr-maas",
        document=document,
        optional_params={},
        headers={},
    )

    source_key: Final = "image_url" if document["type"] == "image_url" else "document_url"
    data: Final = cast(dict[str, object], request.data)
    messages: Final = cast(list[dict[str, object]], data["messages"])
    content: Final = cast(list[dict[str, object]], messages[0]["content"])
    assert data["model"] == "deepseek-ai/deepseek-ocr-maas"
    assert content == [{"type": "image_url", "image_url": document[source_key]}]


@pytest.mark.parametrize(
    "sdk_input",
    (
        ReductoParseV3SdkInput(model="reducto/parse-v3", document=_reducto_document()),
        ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=_reducto_document()),
    ),
)
def test_reducto_parity_cases_are_non_strict_xfails(
    sdk_input: ReductoParseV3SdkInput | ReductoParseLegacySdkInput,
) -> None:
    marks: Final = ocr_fixture_marks(OcrParityCase(litellm_input=sdk_input, provider_responses=()))

    assert len(marks) == 1
    assert marks[0].mark.name == "xfail"
    assert marks[0].mark.kwargs["strict"] is False


def test_supported_parity_cases_have_no_marks() -> None:
    sdk_input: Final = _mistral_input()

    assert ocr_fixture_marks(OcrParityCase(litellm_input=sdk_input, provider_responses=())) == ()


def test_reducto_v3_preserves_nested_provider_params() -> None:
    sdk_input: Final = ReductoParseV3SdkInput(
        model="reducto/parse-v3",
        document=_reducto_document(),
        formatting=ReductoFormatting(table_output_format="html", include=["hyperlinks"]),
        retrieval=ReductoRetrieval(chunking=ReductoChunking(chunk_mode="variable", chunk_size=250, chunk_overlap=32)),
        settings=ReductoSettings(embed_pdf_metadata=True, embed_pdf_metadata_dpi=250, page_range=[1, 3]),
    )

    assert sdk_input.as_sdk_kwargs()["formatting"] == {
        "table_output_format": "html",
        "include": ["hyperlinks"],
    }
    assert sdk_input.as_sdk_kwargs()["retrieval"] == {
        "chunking": {"chunk_mode": "variable", "chunk_size": 250, "chunk_overlap": 32}
    }
    assert sdk_input.as_sdk_kwargs()["settings"] == {
        "embed_pdf_metadata": True,
        "embed_pdf_metadata_dpi": 250,
        "page_range": [1, 3],
    }


def test_reducto_optional_objects_reject_explicit_null() -> None:
    with pytest.raises(ValidationError):
        ReductoParseV3SdkInput.model_validate(
            {
                "model": "reducto/parse-v3",
                "document": _reducto_document(),
                "formatting": None,
            }
        )


@pytest.mark.parametrize(
    "source",
    (
        "https://example.com/document.pdf",
        "not-a-document",
        "data:application/pdf,not-base64",
        "data:application/pdf;base64,not!base64",
    ),
)
def test_reducto_document_rejects_unsupported_sources(source: str) -> None:
    with pytest.raises(ValidationError):
        ReductoDocumentUrlDocument(type="document_url", document_url=source)


def test_reducto_nested_constraints() -> None:
    with pytest.raises(ValidationError, match="less than chunk_size"):
        ReductoChunking(chunk_mode="variable", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError, match="greater than or equal to start"):
        ReductoPageRange(start=3, end=2)
    with pytest.raises(ValidationError):
        ReductoSettings(embed_pdf_metadata_dpi=49)
    with pytest.raises(ValidationError, match="must be unique"):
        ReductoFormatting(include=["hyperlinks", "hyperlinks"])


@settings(max_examples=100, deadline=None)
@given(model=st.sampled_from(MISTRAL_MODELS), data=st.data())
def test_mistral_strategy_only_generates_bounded_valid_sdk_inputs(model: str, data: DataObject) -> None:
    sdk_input: Final = data.draw(mistral_input_strategy(model, INLINE_IMAGE_DATA_URI))
    assert MistralOcrSdkInput.model_validate(sdk_input.canonical_input()) == sdk_input
    optional_fields: Final = frozenset(sdk_input.model_fields_set) - {"model", "document"}
    assert optional_fields in _MISTRAL_OPTION_GROUPS
    if sdk_input.pages is not None:
        assert sdk_input.pages in ([0], [0, 1], "0-2")
    if sdk_input.image_limit is not None:
        assert sdk_input.image_limit == 1
    if sdk_input.image_min_size is not None:
        assert sdk_input.image_min_size == 300
    if sdk_input.table_format is not None:
        assert sdk_input.table_format in {"markdown", "html"}
    if sdk_input.confidence_scores_granularity is not None:
        assert sdk_input.confidence_scores_granularity in {"page", "word", "block"}
    if sdk_input.confidence_scores_granularity == "block":
        assert sdk_input.include_blocks is True
    if model not in _MISTRAL_2512_OR_NEWER:
        assert optional_fields.isdisjoint({"extract_header", "extract_footer", "table_format"})
    if model not in _MISTRAL_4_OR_NEWER:
        assert "include_blocks" not in optional_fields
        assert not isinstance(sdk_input.pages, str)
    if optional_fields:
        assert isinstance(sdk_input.document, DocumentUrlDocument)
        assert sdk_input.document.document_url == structured_pdf_data_uri()


@pytest.mark.parametrize(
    "transport",
    (
        ("image_url", "remote"),
        ("image_url", "data"),
        ("document_url", "remote"),
        ("document_url", "data"),
    ),
)
def test_mistral_strategy_reaches_every_document_transform_branch(transport: tuple[str, str]) -> None:
    sdk_input: Final = _find_fixture(
        mistral_input_strategy("mistral/mistral-ocr-4-1", INLINE_IMAGE_DATA_URI),
        lambda candidate: _document_transport(candidate.document) == transport,
    )

    assert _document_transport(sdk_input.document) == transport


@settings(max_examples=100, deadline=None)
@given(sdk_input=mistral_input_strategy("mistral/mistral-ocr-4-1", INLINE_IMAGE_DATA_URI))
def test_mistral_strategy_values_survive_the_request_transform(sdk_input: MistralOcrSdkInput) -> None:
    sdk_kwargs: Final = sdk_input.as_sdk_kwargs()
    model: Final = cast(str, sdk_kwargs["model"])
    document: Final = cast(DocumentType, sdk_kwargs["document"])
    optional_params: Final = {name: value for name, value in sdk_kwargs.items() if name not in {"model", "document"}}
    config: Final = MistralOCRConfig()
    map_params: Final = cast(_MapOcrParams, config.map_ocr_params)
    transform_request: Final = cast(_TransformOcrRequest, config.transform_ocr_request)
    mapped: Final = map_params(optional_params, {}, model)
    request: Final = transform_request(model, document, mapped, {})
    request_data: Final = cast(dict[str, object], request.data)

    assert mapped == optional_params
    assert request_data == {"model": model, "document": document, **optional_params}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pages", [0]),
        ("pages", [0, 1]),
        ("pages", "0-2"),
        ("include_image_base64", False),
        ("include_image_base64", True),
        ("image_limit", 1),
        ("image_min_size", 300),
        ("extract_header", False),
        ("extract_header", True),
        ("extract_footer", False),
        ("extract_footer", True),
        ("table_format", "markdown"),
        ("table_format", "html"),
        ("confidence_scores_granularity", "page"),
        ("confidence_scores_granularity", "word"),
        ("confidence_scores_granularity", "block"),
        ("include_blocks", False),
        ("include_blocks", True),
    ),
)
def test_mistral_strategy_reaches_every_finite_scalar_value(field: str, value: object) -> None:
    sdk_input: Final = _find_fixture(
        mistral_input_strategy("mistral/mistral-ocr-4-1", INLINE_IMAGE_DATA_URI),
        lambda candidate: field in candidate.model_fields_set and getattr(candidate, field) == value,
    )

    assert getattr(sdk_input, field) == value


@settings(max_examples=50, deadline=None)
@given(sdk_input=reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI))
def test_reducto_v3_strategy_only_generates_bounded_valid_sdk_inputs(sdk_input: ReductoParseV3SdkInput) -> None:
    assert ReductoParseV3SdkInput.model_validate(sdk_input.canonical_input()) == sdk_input
    option_groups: Final = frozenset(sdk_input.model_fields_set) & {"formatting", "retrieval", "settings"}
    assert len(option_groups) <= 1
    if "formatting" in option_groups:
        formatting_fields: Final = frozenset(sdk_input.formatting.model_fields_set)
        assert len(formatting_fields) == 1
        if "table_output_format" in formatting_fields:
            assert sdk_input.formatting.table_output_format in {"dynamic", "html", "md", "json", "csv", "jsonbbox"}
        if "add_page_markers" in formatting_fields:
            assert sdk_input.formatting.add_page_markers in {False, True}
        if "merge_tables" in formatting_fields:
            assert sdk_input.formatting.merge_tables in {False, True}
        if "include" in formatting_fields:
            assert tuple(sdk_input.formatting.include) in _REDUCTO_FORMATTING_INCLUDE_GROUPS
    if "retrieval" in option_groups:
        retrieval_fields: Final = frozenset(sdk_input.retrieval.model_fields_set)
        assert retrieval_fields in {
            frozenset({"chunking"}),
            frozenset({"filter_blocks"}),
            frozenset({"chunking", "embedding_optimized"}),
        }
        chunking: Final = sdk_input.retrieval.chunking
        if "chunking" in retrieval_fields:
            assert chunking.chunk_mode in {"variable", "section", "page", "disabled", "block", "page_sections"}
            assert chunking.chunk_size in {None, 250, 1000, 1500}
            assert chunking.chunk_overlap in {0, 32, 128}
        if chunking.chunk_size is not None or chunking.chunk_overlap:
            assert chunking.chunk_mode == "variable"
        if chunking.chunk_overlap:
            assert chunking.chunk_size == 1000
        if "filter_blocks" in retrieval_fields:
            assert tuple(sdk_input.retrieval.filter_blocks) in _REDUCTO_FILTER_BLOCK_GROUPS
        if "embedding_optimized" in retrieval_fields:
            assert chunking.chunk_mode == "variable"
            assert chunking.chunk_size is None
            assert chunking.chunk_overlap == 0
            assert sdk_input.retrieval.embedding_optimized in {False, True}
    if "settings" in option_groups:
        settings_fields: Final = frozenset(sdk_input.settings.model_fields_set)
        assert settings_fields in {
            frozenset({"model"}),
            frozenset({"ocr_system"}),
            frozenset({"extraction_mode"}),
            frozenset({"return_ocr_data"}),
            frozenset({"return_images"}),
            frozenset({"embed_pdf_metadata"}),
            frozenset({"embed_pdf_metadata", "embed_pdf_metadata_dpi"}),
            frozenset({"timeout"}),
            frozenset({"page_range"}),
        }
        assert settings_fields.isdisjoint(
            {
                "force_url_result",
                "force_file_extension",
                "persist_results",
                "tenant_throttling",
                "document_password",
                "hybrid_vpc",
            }
        )
        if "model" in settings_fields:
            assert sdk_input.settings.model == "r-1"
        if "ocr_system" in settings_fields:
            assert sdk_input.settings.ocr_system in {"standard", "legacy"}
        if "extraction_mode" in settings_fields:
            assert sdk_input.settings.extraction_mode in {"hybrid", "ocr", "metadata"}
        if "return_ocr_data" in settings_fields:
            assert sdk_input.settings.return_ocr_data is True
        if "return_images" in settings_fields:
            assert tuple(sdk_input.settings.return_images) in _REDUCTO_RETURN_IMAGE_GROUPS
        if "embed_pdf_metadata_dpi" in settings_fields:
            assert sdk_input.settings.embed_pdf_metadata is True
            assert sdk_input.settings.embed_pdf_metadata_dpi in {50, 100, 250}
        if "timeout" in settings_fields:
            assert sdk_input.settings.timeout == 300.0
        if sdk_input.settings.page_range is not None:
            dumped_range: Final = cast(
                dict[str, object], sdk_input.settings.model_dump(mode="json", exclude_unset=True)
            )["page_range"]
            assert dumped_range in (
                {"start": 1, "end": 1},
                {"start": 1, "end": 3},
                [{"start": 1, "end": 2}, {"start": 4, "end": 5}],
            )


@settings(max_examples=60, deadline=None)
@given(sdk_input=reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI))
def test_reducto_v3_strategy_values_survive_the_request_transform(sdk_input: ReductoParseV3SdkInput) -> None:
    sdk_kwargs: Final = sdk_input.as_sdk_kwargs()
    model: Final = cast(str, sdk_kwargs["model"])
    document: Final = cast(DocumentType, sdk_kwargs["document"])
    optional_params: Final = {
        name: value for name, value in sdk_kwargs.items() if name not in {"model", "document", "custom_llm_provider"}
    }
    config: Final = ReductoParseV3Config()
    map_params: Final = cast(_MapOcrParams, config.map_ocr_params)
    transform_request: Final = cast(_TransformOcrRequest, config.transform_ocr_request)
    mapped: Final = map_params(optional_params, {}, model)

    with patch.object(config, "_ensure_file_id_sync", return_value="reducto://fixture-document.pdf"):
        request: Final = transform_request(model, document, mapped, {})

    assert mapped == optional_params
    assert cast(dict[str, object], request.data) == {
        "input": "reducto://fixture-document.pdf",
        **optional_params,
    }


def test_reducto_v3_strategy_reaches_image_upload_branch_without_options() -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: isinstance(candidate.document, ReductoImageUrlDocument),
    )

    assert isinstance(sdk_input.document, ReductoImageUrlDocument)
    assert sdk_input.document.image_url.startswith("data:image/")
    assert sdk_input.model_fields_set == {"model", "document"}


@pytest.mark.parametrize(
    ("model", "provider"),
    (("reducto/parse-v3", None), ("parse-v3", "reducto")),
)
def test_reducto_v3_strategy_reaches_every_routing_form(model: str, provider: str | None) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: candidate.model == model and candidate.custom_llm_provider == provider,
    )

    assert sdk_input.model == model
    assert sdk_input.custom_llm_provider == provider


@pytest.mark.parametrize("table_format", ("dynamic", "html", "md", "json", "csv", "jsonbbox"))
def test_reducto_v3_strategy_reaches_every_table_format(table_format: str) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "formatting" in candidate.model_fields_set
            and "table_output_format" in candidate.formatting.model_fields_set
            and candidate.formatting.table_output_format == table_format
        ),
    )

    assert sdk_input.formatting.table_output_format == table_format


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("add_page_markers", False),
        ("add_page_markers", True),
        ("merge_tables", False),
        ("merge_tables", True),
    ),
)
def test_reducto_v3_strategy_reaches_every_formatting_boolean(field: str, value: bool) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "formatting" in candidate.model_fields_set
            and field in candidate.formatting.model_fields_set
            and getattr(candidate.formatting, field) is value
        ),
    )

    assert getattr(sdk_input.formatting, field) is value


@pytest.mark.parametrize("include", _REDUCTO_FORMATTING_INCLUDE_GROUPS)
def test_reducto_v3_strategy_reaches_every_formatting_include(include: tuple[str, ...]) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "formatting" in candidate.model_fields_set
            and "include" in candidate.formatting.model_fields_set
            and tuple(candidate.formatting.include) == include
        ),
    )

    assert tuple(sdk_input.formatting.include) == include


@pytest.mark.parametrize("chunk_mode", ("variable", "section", "page", "disabled", "block", "page_sections"))
def test_reducto_v3_strategy_reaches_every_chunk_mode(chunk_mode: str) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "retrieval" in candidate.model_fields_set
            and "chunking" in candidate.retrieval.model_fields_set
            and candidate.retrieval.chunking.chunk_mode == chunk_mode
        ),
    )

    assert sdk_input.retrieval.chunking.chunk_mode == chunk_mode


@pytest.mark.parametrize("chunk_size", (250, 1000, 1500))
def test_reducto_v3_strategy_reaches_every_chunk_size(chunk_size: int) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: candidate.retrieval.chunking.chunk_size == chunk_size,
    )

    assert sdk_input.retrieval.chunking.chunk_mode == "variable"
    assert sdk_input.retrieval.chunking.chunk_size == chunk_size


@pytest.mark.parametrize("chunk_overlap", (32, 128))
def test_reducto_v3_strategy_reaches_every_chunk_overlap(chunk_overlap: int) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: candidate.retrieval.chunking.chunk_overlap == chunk_overlap,
    )

    assert sdk_input.retrieval.chunking.chunk_mode == "variable"
    assert sdk_input.retrieval.chunking.chunk_size == 1000
    assert sdk_input.retrieval.chunking.chunk_overlap == chunk_overlap


@pytest.mark.parametrize("filter_blocks", _REDUCTO_FILTER_BLOCK_GROUPS)
def test_reducto_v3_strategy_reaches_every_filter_block_group(filter_blocks: tuple[str, ...]) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "retrieval" in candidate.model_fields_set
            and "filter_blocks" in candidate.retrieval.model_fields_set
            and tuple(candidate.retrieval.filter_blocks) == filter_blocks
        ),
    )

    assert tuple(sdk_input.retrieval.filter_blocks) == filter_blocks


@pytest.mark.parametrize("embedding_optimized", (False, True))
def test_reducto_v3_strategy_reaches_every_embedding_setting(embedding_optimized: bool) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "retrieval" in candidate.model_fields_set
            and "embedding_optimized" in candidate.retrieval.model_fields_set
            and candidate.retrieval.embedding_optimized is embedding_optimized
        ),
    )

    assert sdk_input.retrieval.chunking.chunk_mode == "variable"
    assert sdk_input.retrieval.embedding_optimized is embedding_optimized


@pytest.mark.parametrize("dpi", (50, 100, 250))
def test_reducto_v3_strategy_reaches_every_metadata_dpi(dpi: int) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "settings" in candidate.model_fields_set
            and "embed_pdf_metadata_dpi" in candidate.settings.model_fields_set
            and candidate.settings.embed_pdf_metadata_dpi == dpi
        ),
    )

    assert sdk_input.settings.embed_pdf_metadata is True
    assert sdk_input.settings.embed_pdf_metadata_dpi == dpi


def test_reducto_v3_strategy_reaches_metadata_with_default_dpi_omitted() -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "settings" in candidate.model_fields_set and candidate.settings.model_fields_set == {"embed_pdf_metadata"}
        ),
    )

    assert sdk_input.settings.embed_pdf_metadata is True
    assert "embed_pdf_metadata_dpi" not in sdk_input.settings.model_fields_set


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", "r-1"),
        ("ocr_system", "standard"),
        ("ocr_system", "legacy"),
        ("extraction_mode", "hybrid"),
        ("extraction_mode", "ocr"),
        ("extraction_mode", "metadata"),
        ("return_ocr_data", True),
        ("timeout", 300.0),
    ),
)
def test_reducto_v3_strategy_reaches_every_scalar_setting(field: str, value: object) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "settings" in candidate.model_fields_set
            and field in candidate.settings.model_fields_set
            and getattr(candidate.settings, field) == value
        ),
    )

    assert getattr(sdk_input.settings, field) == value


@pytest.mark.parametrize("return_images", _REDUCTO_RETURN_IMAGE_GROUPS)
def test_reducto_v3_strategy_reaches_every_return_image_group(return_images: tuple[str, ...]) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            "settings" in candidate.model_fields_set
            and "return_images" in candidate.settings.model_fields_set
            and tuple(candidate.settings.return_images) == return_images
        ),
    )

    assert tuple(sdk_input.settings.return_images) == return_images


@pytest.mark.parametrize(
    "page_range",
    (
        {"start": 1, "end": 1},
        {"start": 1, "end": 3},
        [{"start": 1, "end": 2}, {"start": 4, "end": 5}],
    ),
)
def test_reducto_v3_strategy_reaches_every_page_range_shape(page_range: object) -> None:
    sdk_input: Final = _find_fixture(
        reducto_v3_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            cast(
                dict[str, object],
                candidate.settings.model_dump(mode="json", exclude_unset=True),
            ).get("page_range")
            == page_range
        ),
    )

    assert sdk_input.settings.model_dump(mode="json", exclude_unset=True)["page_range"] == page_range


@settings(max_examples=10, deadline=None)
@given(sdk_input=reducto_legacy_input_strategy())
def test_reducto_legacy_strategy_generates_valid_litellm_inputs(sdk_input: ReductoParseLegacySdkInput) -> None:
    assert ReductoParseLegacySdkInput.model_validate(sdk_input.canonical_input()) == sdk_input
    assert "enhance" not in sdk_input.model_fields_set


@settings(max_examples=10, deadline=None)
@given(sdk_input=reducto_legacy_input_strategy())
def test_reducto_legacy_strategy_values_survive_the_request_transform(
    sdk_input: ReductoParseLegacySdkInput,
) -> None:
    sdk_kwargs: Final = sdk_input.as_sdk_kwargs()
    model: Final = cast(str, sdk_kwargs["model"])
    document: Final = cast(DocumentType, sdk_kwargs["document"])
    config: Final = ReductoParseLegacyConfig()
    transform_request: Final = cast(_TransformOcrRequest, config.transform_ocr_request)

    with patch.object(config, "_ensure_file_id_sync", return_value="reducto://fixture-document.pdf"):
        request: Final = transform_request(model, document, {}, {})

    assert cast(dict[str, object], request.data) == {
        "document_url": "reducto://fixture-document.pdf",
    }


@pytest.mark.parametrize(
    ("model", "provider"),
    (("reducto/parse-legacy", None), ("parse-legacy", "reducto")),
)
def test_reducto_legacy_strategy_reaches_every_routing_form(model: str, provider: str | None) -> None:
    sdk_input: Final = _find_fixture(
        reducto_legacy_input_strategy(),
        lambda candidate: candidate.model == model and candidate.custom_llm_provider == provider,
    )

    assert sdk_input.model == model
    assert sdk_input.custom_llm_provider == provider


@settings(max_examples=50, deadline=None)
@given(sdk_input=azure_mistral_input_strategy(INLINE_IMAGE_DATA_URI))
def test_azure_mistral_strategy_is_contained_to_gateway_capabilities(
    sdk_input: AzureMistralOcrSdkInput,
) -> None:
    optional_fields: Final = frozenset(sdk_input.model_fields_set) - {"model", "document"}

    assert optional_fields in _AZURE_MISTRAL_OPTION_GROUPS
    assert optional_fields.isdisjoint(
        {
            "document_annotation_prompt",
            "extract_header",
            "extract_footer",
            "table_format",
            "include_blocks",
            "id",
        }
    )
    assert not isinstance(sdk_input.pages, str)
    assert sdk_input.confidence_scores_granularity in {None, "page", "word"}
    if optional_fields:
        assert isinstance(sdk_input.document, DocumentUrlDocument)
        assert sdk_input.document.document_url == structured_pdf_data_uri()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pages", [0]),
        ("pages", [0, 1]),
        ("include_image_base64", False),
        ("include_image_base64", True),
        ("image_limit", 1),
        ("image_min_size", 300),
        ("confidence_scores_granularity", "page"),
        ("confidence_scores_granularity", "word"),
    ),
)
def test_azure_mistral_strategy_reaches_every_gateway_scalar(field: str, value: object) -> None:
    sdk_input: Final = _find_fixture(
        azure_mistral_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: field in candidate.model_fields_set and getattr(candidate, field) == value,
    )

    assert getattr(sdk_input, field) == value


@pytest.mark.parametrize("field", ("bbox_annotation_format", "document_annotation_format"))
def test_azure_mistral_strategy_reaches_every_gateway_schema(field: str) -> None:
    sdk_input: Final = _find_fixture(
        azure_mistral_input_strategy(INLINE_IMAGE_DATA_URI),
        lambda candidate: frozenset(candidate.model_fields_set) - {"model", "document"} == frozenset({field}),
    )

    assert frozenset(sdk_input.model_fields_set) - {"model", "document"} == {field}


@settings(max_examples=50, deadline=None)
@given(sdk_input=azure_mistral_input_strategy(INLINE_IMAGE_DATA_URI))
def test_azure_mistral_strategy_exercises_url_conversion_and_inline_bypass(
    sdk_input: AzureMistralOcrSdkInput,
) -> None:
    sdk_kwargs: Final = sdk_input.as_sdk_kwargs()
    model: Final = cast(str, sdk_kwargs["model"])
    document: Final = cast(DocumentType, sdk_kwargs["document"])
    optional_params: Final = {name: value for name, value in sdk_kwargs.items() if name not in {"model", "document"}}
    config: Final = AzureAIOCRConfig()
    map_params: Final = cast(_MapOcrParams, config.map_ocr_params)
    transform_request: Final = cast(_TransformOcrRequest, config.transform_ocr_request)
    mapped: Final = map_params(optional_params, {}, model)

    request: Final = _transform_with_stubbed_download(transform_request, model, document, mapped)

    source_key: Final = "image_url" if document["type"] == "image_url" else "document_url"
    source: Final = document[source_key]
    expected_document: Final = dict(document)
    if not source.startswith("data:"):
        media_type: Final = "image/png" if document["type"] == "image_url" else "application/pdf"
        expected_document[source_key] = f"data:{media_type};base64,AA=="

    assert mapped == optional_params
    assert cast(dict[str, object], request.data) == {
        "model": model,
        "document": expected_document,
        **optional_params,
    }


@settings(max_examples=30, deadline=None)
@given(sdk_input=azure_document_intelligence_input_strategy())
def test_azure_document_intelligence_strategy_only_generates_litellm_inputs(
    sdk_input: AzureDocumentIntelligenceOcrSdkInput,
) -> None:
    assert sdk_input.req_format == "litellm"
    assert sdk_input.model in AZURE_DOCUMENT_INTELLIGENCE_RECORDING_MODELS
    assert "contract" not in sdk_input.as_sdk_kwargs()
    optional_fields: Final = frozenset(sdk_input.model_fields_set) - {"model", "document"}
    assert optional_fields in {
        frozenset[str](),
        frozenset({"pages"}),
        frozenset({"features"}),
        frozenset({"pages", "features"}),
        frozenset({"req_format"}),
    }
    if sdk_input.pages is not None:
        assert sdk_input.pages in ([0], [2, 0, 0, 1], ["1", "2-4"], "1-4, 5", [0, 1])
    if isinstance(sdk_input.features, list):
        assert tuple(sdk_input.features) in {
            ("languages",),
            ("ocrHighResolution",),
            ("barcodes",),
            ("formulas",),
            ("styleFont",),
            ("keyValuePairs",),
            ("languages", "styleFont"),
        }
    if isinstance(sdk_input.features, str):
        assert sdk_input.features == "languages, styleFont"


@settings(max_examples=50, deadline=None)
@given(sdk_input=azure_document_intelligence_input_strategy())
def test_azure_document_intelligence_strategy_exercises_request_transform(
    sdk_input: AzureDocumentIntelligenceOcrSdkInput,
) -> None:
    sdk_kwargs: Final = sdk_input.as_sdk_kwargs()
    model: Final = cast(str, sdk_kwargs["model"])
    document: Final = cast(DocumentType, sdk_kwargs["document"])
    optional_params: Final = {name: value for name, value in sdk_kwargs.items() if name not in {"model", "document"}}
    config: Final = AzureDocumentIntelligenceOCRConfig()
    map_params: Final = cast(_MapOcrParams, config.map_ocr_params)
    get_complete_url: Final = cast(_GetCompleteUrl, config.get_complete_url)
    transform_request: Final = cast(_TransformOcrRequest, config.transform_ocr_request)
    mapped: Final = map_params(optional_params, {}, model)
    url: Final = get_complete_url("https://document.example", model, mapped)
    query: Final = parse_qs(urlparse(url).query)
    request: Final = transform_request(model, document, mapped, {})

    if sdk_input.pages is None:
        assert "pages" not in mapped
        assert "pages" not in query
    else:
        expected_pages: Final = _normalized_azure_pages(sdk_input.pages)
        assert mapped["pages"] == expected_pages
        assert query["pages"] == [expected_pages]
    if sdk_input.features is None:
        assert "features" not in mapped
        assert "features" not in query
    else:
        raw_features: Final = (
            sdk_input.features.split(",") if isinstance(sdk_input.features, str) else sdk_input.features
        )
        expected_features: Final = ",".join(feature.strip() for feature in raw_features)
        assert mapped["features"] == expected_features
        assert query["features"] == [expected_features]

    source: Final = document["document_url"] if document["type"] == "document_url" else document["image_url"]
    assert isinstance(source, str)
    expected_body: Final = (
        {"base64Source": source.partition(",")[2]} if source.startswith("data:") else {"urlSource": source}
    )
    assert cast(dict[str, object], request.data) == expected_body


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pages", [0]),
        ("pages", [2, 0, 0, 1]),
        ("pages", ["1", "2-4"]),
        ("pages", "1-4, 5"),
        ("features", ["languages"]),
        ("features", ["ocrHighResolution"]),
        ("features", ["barcodes"]),
        ("features", ["formulas"]),
        ("features", ["styleFont"]),
        ("features", ["keyValuePairs"]),
        ("features", "languages, styleFont"),
    ),
)
def test_azure_document_intelligence_strategy_reaches_every_finite_value(field: str, value: object) -> None:
    sdk_input: Final = _find_fixture(
        azure_document_intelligence_input_strategy(),
        lambda candidate: field in candidate.model_fields_set and getattr(candidate, field) == value,
    )

    assert getattr(sdk_input, field) == value


def test_azure_document_intelligence_strategy_reaches_combined_query_branch() -> None:
    sdk_input: Final = _find_fixture(
        azure_document_intelligence_input_strategy(),
        lambda candidate: {"pages", "features"}.issubset(candidate.model_fields_set),
    )

    assert sdk_input.pages == [0, 1]
    assert sdk_input.features == ["languages", "styleFont"]


@pytest.mark.parametrize(
    "transport",
    (("document_url", "data"), ("image_url", "remote")),
)
def test_azure_document_intelligence_strategy_reaches_body_source_branches(
    transport: tuple[str, str],
) -> None:
    sdk_input: Final = _find_fixture(
        azure_document_intelligence_input_strategy(),
        lambda candidate: _document_transport(candidate.document) == transport,
    )

    assert _document_transport(sdk_input.document) == transport


@settings(max_examples=50, deadline=None)
@given(sdk_input=vertex_mistral_input_strategy("project-1", "us-central1", INLINE_IMAGE_DATA_URI))
def test_vertex_mistral_strategy_is_contained_to_2505_capabilities(
    sdk_input: VertexMistralOcrSdkInput,
) -> None:
    optional_fields: Final = frozenset(sdk_input.model_fields_set) - {
        "model",
        "document",
        "vertex_project",
        "vertex_location",
    }

    assert optional_fields in _MISTRAL_2505_OPTION_GROUPS
    assert optional_fields.isdisjoint({"extract_header", "extract_footer", "table_format", "include_blocks", "id"})
    assert not isinstance(sdk_input.pages, str)
    assert sdk_input.confidence_scores_granularity in {None, "page", "word"}
    if optional_fields:
        assert isinstance(sdk_input.document, DocumentUrlDocument)
        assert sdk_input.document.document_url == structured_pdf_data_uri()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pages", [0]),
        ("pages", [0, 1]),
        ("include_image_base64", False),
        ("include_image_base64", True),
        ("image_limit", 1),
        ("image_min_size", 300),
        ("confidence_scores_granularity", "page"),
        ("confidence_scores_granularity", "word"),
    ),
)
def test_vertex_mistral_strategy_reaches_every_2505_scalar(field: str, value: object) -> None:
    sdk_input: Final = _find_fixture(
        vertex_mistral_input_strategy("project-1", "us-central1", INLINE_IMAGE_DATA_URI),
        lambda candidate: field in candidate.model_fields_set and getattr(candidate, field) == value,
    )

    assert getattr(sdk_input, field) == value


@pytest.mark.parametrize(
    "fields",
    (
        frozenset({"bbox_annotation_format"}),
        frozenset({"document_annotation_format"}),
        frozenset({"document_annotation_format", "document_annotation_prompt"}),
    ),
)
def test_vertex_mistral_strategy_reaches_every_2505_schema_group(fields: frozenset[str]) -> None:
    sdk_input: Final = _find_fixture(
        vertex_mistral_input_strategy("project-1", "us-central1", INLINE_IMAGE_DATA_URI),
        lambda candidate: (
            frozenset(candidate.model_fields_set) - {"model", "document", "vertex_project", "vertex_location"} == fields
        ),
    )

    assert frozenset(sdk_input.model_fields_set) - {"model", "document", "vertex_project", "vertex_location"} == fields


@settings(max_examples=50, deadline=None)
@given(sdk_input=vertex_mistral_input_strategy("project-1", "us-central1", INLINE_IMAGE_DATA_URI))
def test_vertex_mistral_strategy_exercises_url_conversion_and_inline_bypass(
    sdk_input: VertexMistralOcrSdkInput,
) -> None:
    sdk_kwargs: Final = sdk_input.as_sdk_kwargs()
    model: Final = cast(str, sdk_kwargs["model"])
    document: Final = cast(DocumentType, sdk_kwargs["document"])
    optional_params: Final = {
        name: value
        for name, value in sdk_kwargs.items()
        if name not in {"model", "document", "vertex_project", "vertex_location"}
    }
    config: Final = VertexAIOCRConfig()
    map_params: Final = cast(_MapOcrParams, config.map_ocr_params)
    transform_request: Final = cast(_TransformOcrRequest, config.transform_ocr_request)
    mapped: Final = map_params(optional_params, {}, model)

    request: Final = _transform_with_stubbed_download(transform_request, model, document, mapped)

    source_key: Final = "image_url" if document["type"] == "image_url" else "document_url"
    source: Final = document[source_key]
    expected_document: Final = dict(document)
    if not source.startswith("data:"):
        media_type: Final = "image/png" if document["type"] == "image_url" else "application/pdf"
        expected_document[source_key] = f"data:{media_type};base64,AA=="

    assert mapped == optional_params
    assert cast(dict[str, object], request.data) == {
        "model": model,
        "document": expected_document,
        **optional_params,
    }


@settings(max_examples=30, deadline=None)
@given(sdk_input=vertex_deepseek_input_strategy("project-1", "us-central1", INLINE_IMAGE_DATA_URI))
def test_vertex_deepseek_strategy_only_generates_litellm_inputs(
    sdk_input: VertexDeepSeekOcrSdkInput,
) -> None:
    assert sdk_input.vertex_project == "project-1"
    assert "contract" not in sdk_input.as_sdk_kwargs()
    assert _document_transport(sdk_input.document) == ("image_url", "data")


def test_vertex_deepseek_strategy_reaches_documented_image_branch() -> None:
    sdk_input: Final = _find_fixture(
        vertex_deepseek_input_strategy("project-1", "us-central1", INLINE_IMAGE_DATA_URI),
        lambda candidate: _document_transport(candidate.document) == ("image_url", "data"),
    )

    assert _document_transport(sdk_input.document) == ("image_url", "data")
