from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from pydantic import Field, field_validator

from tests.route_parity.fixtures.recording import ProviderSpec
from tests.test_litellm.ocr.fixtures.base import OcrDocument, OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrRecordingTarget,
    invoke_with_api_key,
    parameter_strategy,
    pdf_document,
    public_document_strategy,
    sampled_list_strategy,
    sampled_scalar_strategy,
)
from tests.test_litellm.ocr.fixtures.mistral import (
    MISTRAL_MODEL,
    MistralCompatibleOcrSdkInput,
    MistralOcrSdkInput,
    mistral_input_strategy,
)

AzureMistralModel = Literal["azure_ai/mistral-document-ai-2512",]
AzureDocumentIntelligenceModel = Literal[
    "azure_ai/doc-intelligence/prebuilt-read",
    "azure_ai/doc-intelligence/prebuilt-layout",
    "azure_ai/doc-intelligence/prebuilt-document",
]

AZURE_MISTRAL_MODELS: Final[tuple[AzureMistralModel, ...]] = ("azure_ai/mistral-document-ai-2512",)
AZURE_DOCUMENT_INTELLIGENCE_MODELS: Final[tuple[AzureDocumentIntelligenceModel, ...]] = (
    "azure_ai/doc-intelligence/prebuilt-read",
    "azure_ai/doc-intelligence/prebuilt-layout",
    "azure_ai/doc-intelligence/prebuilt-document",
)


class AzureMistralOcrSdkInput(MistralCompatibleOcrSdkInput):
    boundary: str = Field(default="azure_mistral", pattern=r"^azure_mistral$")
    model: AzureMistralModel
    custom_llm_provider: Literal["azure_ai"] | None = None

    @field_validator("model")
    @classmethod
    def validate_model_namespace(cls, model: str) -> str:
        if not model.startswith("azure_ai/"):
            raise ValueError("Azure Mistral models must use the azure_ai/ LiteLLM namespace")
        return model


class AzureDocumentIntelligenceOcrSdkInput(OcrSdkInputBase):
    boundary: str = Field(default="azure_document_intelligence", pattern=r"^azure_document_intelligence$")
    model: AzureDocumentIntelligenceModel
    document: OcrDocument
    custom_llm_provider: Literal["azure_ai"] | None = None
    pages: str | list[int] | None = None
    features: str | list[str] | None = None
    req_format: Literal["litellm"] = "litellm"


def _as_azure_mistral(case_input: MistralOcrSdkInput, model: AzureMistralModel) -> AzureMistralOcrSdkInput:
    values: Final = case_input.model_dump(
        mode="python",
        exclude={"boundary", "model", "custom_llm_provider"},
        exclude_unset=True,
    )
    return AzureMistralOcrSdkInput.model_validate({**values, "model": model})


_AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL: Final[AzureDocumentIntelligenceModel] = (
    "azure_ai/doc-intelligence/prebuilt-layout"
)
_AZURE_DOCUMENT_INTELLIGENCE_DOCUMENT_MODEL: Final[AzureDocumentIntelligenceModel] = (
    "azure_ai/doc-intelligence/prebuilt-document"
)


def _document_intelligence_input(
    model: AzureDocumentIntelligenceModel,
    document: OcrDocument,
    optional_params: dict[str, object] | None = None,
) -> AzureDocumentIntelligenceOcrSdkInput:
    return AzureDocumentIntelligenceOcrSdkInput.model_validate(
        {"model": model, "document": document, **(optional_params or {})}
    )


def azure_document_intelligence_input_strategy() -> SearchStrategy[AzureDocumentIntelligenceOcrSdkInput]:
    document: Final = pdf_document()
    pages: Final = st.one_of(
        parameter_strategy("pages", sampled_list_strategy(((0,), (0, 1)))),
        parameter_strategy("pages", sampled_scalar_strategy(("1", "1,2", "1-2"))),
    )
    common_features: Final = parameter_strategy(
        "features",
        st.one_of(
            sampled_list_strategy(
                (("languages",), ("ocrHighResolution",), ("barcodes",), ("formulas",), ("styleFont",))
            ),
            sampled_scalar_strategy(("languages,styleFont",)),
        ),
    )
    return st.one_of(
        st.sampled_from(AZURE_DOCUMENT_INTELLIGENCE_MODELS).map(
            lambda model: _document_intelligence_input(model, document)
        ),
        public_document_strategy().map(
            lambda selected_document: _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL, selected_document
            )
        ),
        pages.map(
            lambda optional_params: _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL, document, optional_params
            )
        ),
        common_features.map(
            lambda optional_params: _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL, document, optional_params
            )
        ),
        st.just(
            _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_DOCUMENT_MODEL,
                document,
                {"features": ["keyValuePairs"]},
            )
        ),
        st.just(
            _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL,
                document,
                {"req_format": "litellm"},
            )
        ),
    )


def azure_mistral_recording_targets(
    environ: Mapping[str, str], client: OcrFixtureClient
) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("AZURE_AI_API_KEY")
    upstream_base: Final = environ.get("AZURE_AI_API_BASE")
    if not api_key or not upstream_base:
        return ()
    return (
        OcrRecordingTarget(
            name="azure-mistral",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                st.sampled_from(AZURE_MISTRAL_MODELS).flatmap(
                    lambda model: mistral_input_strategy(MISTRAL_MODEL, feature_level="2512").map(
                        lambda case_input: _as_azure_mistral(case_input, model)
                    )
                ),
            ),
            invocation=invoke_with_api_key(client, api_key),
        ),
    )


def azure_document_intelligence_recording_targets(
    environ: Mapping[str, str], client: OcrFixtureClient
) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    upstream_base: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not api_key or not upstream_base:
        return ()
    return (
        OcrRecordingTarget(
            name="azure-document-intelligence",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(SearchStrategy[OcrSdkInputBase], azure_document_intelligence_input_strategy()),
            invocation=invoke_with_api_key(client, api_key),
        ),
    )
