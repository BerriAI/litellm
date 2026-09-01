from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final, cast

import pytest
from hypothesis import given, settings
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.llms.azure_ai.ocr.document_intelligence.transformation import AzureDocumentIntelligenceOCRConfig
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig
from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.llms.reducto.ocr.transformation import ReductoParseLegacyConfig, ReductoParseV3Config
from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig
from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig
from tests.test_litellm.ocr.conftest import ocr_fixture_marks
from tests.test_litellm.ocr.fixtures.azure import (
    AZURE_DOCUMENT_INTELLIGENCE_MODELS,
    AZURE_MISTRAL_MODELS,
    AzureDocumentIntelligenceOcrSdkInput,
    AzureMistralOcrSdkInput,
    azure_document_intelligence_input_strategy,
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
from tests.test_litellm.ocr.fixtures.models import OcrParityCase
from tests.test_litellm.ocr.fixtures.reducto import (
    REDUCTO_LEGACY_MODELS,
    REDUCTO_V3_MODELS,
    ReductoChunking,
    ReductoDocumentUrlDocument,
    ReductoFormatting,
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
)

COMMON_FIELDS: Final = frozenset(
    {"boundary", "model", "document", "custom_llm_provider", "vertex_project", "vertex_location"}
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
def test_provider_boundary_is_explicit_but_not_forwarded(sdk_input: OcrSdkInputBase) -> None:
    assert sdk_input.canonical_input()["boundary"] == sdk_input.boundary
    assert "boundary" not in sdk_input.as_sdk_kwargs()


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


@pytest.mark.parametrize("model", ("deepseek-ocr-maas", "deepseek-ai/deepseek-ocr-maas"))
def test_vertex_deepseek_request_uses_single_provider_namespace(model: str) -> None:
    request: Final = VertexAIDeepSeekOCRConfig().transform_ocr_request(  # pyright: ignore[reportUnknownMemberType]
        model=model,
        document={"type": "image_url", "image_url": "data:image/png;base64,AA=="},
        optional_params={},
        headers={},
    )

    data: Final = cast(dict[str, object], request.data)
    assert data["model"] == "deepseek-ai/deepseek-ocr-maas"


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


@settings(max_examples=50, deadline=None)
@given(sdk_input=mistral_input_strategy("mistral/mistral-ocr-4-1"))
def test_mistral_strategy_only_generates_valid_sdk_inputs(sdk_input: MistralOcrSdkInput) -> None:
    assert MistralOcrSdkInput.model_validate(sdk_input.canonical_input()) == sdk_input


@settings(max_examples=50, deadline=None)
@given(sdk_input=reducto_v3_input_strategy())
def test_reducto_v3_strategy_only_generates_valid_sdk_inputs(sdk_input: ReductoParseV3SdkInput) -> None:
    assert ReductoParseV3SdkInput.model_validate(sdk_input.canonical_input()) == sdk_input


@settings(max_examples=10, deadline=None)
@given(sdk_input=reducto_legacy_input_strategy())
def test_reducto_legacy_strategy_generates_valid_litellm_inputs(sdk_input: ReductoParseLegacySdkInput) -> None:
    assert ReductoParseLegacySdkInput.model_validate(sdk_input.canonical_input()) == sdk_input


@settings(max_examples=30, deadline=None)
@given(sdk_input=azure_document_intelligence_input_strategy())
def test_azure_document_intelligence_strategy_only_generates_litellm_inputs(
    sdk_input: AzureDocumentIntelligenceOcrSdkInput,
) -> None:
    assert sdk_input.req_format == "litellm"
    assert "boundary" not in sdk_input.as_sdk_kwargs()


@settings(max_examples=30, deadline=None)
@given(sdk_input=vertex_deepseek_input_strategy("project-1", "us-central1"))
def test_vertex_deepseek_strategy_only_generates_litellm_inputs(
    sdk_input: VertexDeepSeekOcrSdkInput,
) -> None:
    assert sdk_input.vertex_project == "project-1"
    assert "boundary" not in sdk_input.as_sdk_kwargs()
