from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from pydantic import StrictInt, StrictStr, field_validator

from ......shared.parity.fixtures.recording import UpstreamEndpoint
from .base import OcrDocument, OcrSdkInputBase
from .common import (
    OcrFixtureClient,
    OcrRecordingTarget,
    image_document,
    invoke_with_api_key,
    pdf_document,
)
from .mistral import (
    MistralCompatibleOcrSdkInput,
    mistral_input_values_strategy,
)

AzureMistralModel = Literal["azure_ai/mistral-document-ai-2512",]
AzureMistralFixtureModel = AzureMistralModel | Literal["azure_ai/invalid-ocr-model-for-parity"]
AzureDocumentIntelligenceModel = Literal[
    "azure_ai/doc-intelligence/prebuilt-read",
    "azure_ai/doc-intelligence/prebuilt-layout",
    "azure_ai/doc-intelligence/prebuilt-document",
]
AzureDocumentIntelligenceFixtureModel = (
    AzureDocumentIntelligenceModel | Literal["azure_ai/doc-intelligence/invalid-ocr-model-for-parity"]
)

AZURE_MISTRAL_MODELS: Final[tuple[AzureMistralModel, ...]] = ("azure_ai/mistral-document-ai-2512",)
AZURE_DOCUMENT_INTELLIGENCE_MODELS: Final[tuple[AzureDocumentIntelligenceModel, ...]] = (
    "azure_ai/doc-intelligence/prebuilt-read",
    "azure_ai/doc-intelligence/prebuilt-layout",
    "azure_ai/doc-intelligence/prebuilt-document",
)
# API v4 replaces prebuilt-document with prebuilt-layout plus keyValuePairs. Keep
# the broader fixture model above so existing recordings remain loadable.
AZURE_DOCUMENT_INTELLIGENCE_RECORDING_MODELS: Final[tuple[AzureDocumentIntelligenceModel, ...]] = (
    "azure_ai/doc-intelligence/prebuilt-read",
    "azure_ai/doc-intelligence/prebuilt-layout",
)


class AzureMistralOcrSdkInput(MistralCompatibleOcrSdkInput):
    contract: Literal["azure_mistral"] = "azure_mistral"
    model: AzureMistralFixtureModel
    custom_llm_provider: Literal["azure_ai"] | None = None

    @field_validator("model")
    @classmethod
    def validate_model_namespace(cls, model: str) -> str:
        if not model.startswith("azure_ai/"):
            raise ValueError("Azure Mistral models must use the azure_ai/ LiteLLM namespace")
        return model


class AzureDocumentIntelligenceOcrSdkInput(OcrSdkInputBase):
    contract: Literal["azure_document_intelligence"] = "azure_document_intelligence"
    model: AzureDocumentIntelligenceFixtureModel
    document: OcrDocument
    custom_llm_provider: Literal["azure_ai"] | None = None
    pages: str | list[StrictInt] | list[StrictStr] | None = None
    features: str | list[str] | None = None
    req_format: Literal["litellm"] = "litellm"


AZURE_MISTRAL_PROVIDER_REJECTED_INPUTS: Final[tuple[AzureMistralOcrSdkInput, ...]] = (
    AzureMistralOcrSdkInput(
        model="azure_ai/invalid-ocr-model-for-parity",
        document=pdf_document(),
    ),
)
AZURE_DOCUMENT_INTELLIGENCE_PROVIDER_REJECTED_INPUTS: Final[tuple[AzureDocumentIntelligenceOcrSdkInput, ...]] = (
    AzureDocumentIntelligenceOcrSdkInput(
        model="azure_ai/doc-intelligence/invalid-ocr-model-for-parity",
        document=pdf_document(),
    ),
)


def _azure_mistral_input(values: dict[str, object], model: AzureMistralModel) -> AzureMistralOcrSdkInput:
    return AzureMistralOcrSdkInput.model_validate({**values, "model": model})


def azure_mistral_input_strategy(inline_image_data_uri: str) -> SearchStrategy[AzureMistralOcrSdkInput]:
    # Foundry's active gateway schema rejects 2512-only controls and
    # document_annotation_prompt, even though native Mistral accepts them.
    return st.builds(
        _azure_mistral_input,
        values=mistral_input_values_strategy("2505", inline_image_data_uri, include_document_annotation_prompt=False),
        model=st.sampled_from(AZURE_MISTRAL_MODELS),
    )


_AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL: Final[AzureDocumentIntelligenceModel] = (
    "azure_ai/doc-intelligence/prebuilt-layout"
)


def _document_intelligence_input(
    model: AzureDocumentIntelligenceModel,
    document: OcrDocument,
    optional_params: Mapping[str, object] | None = None,
) -> AzureDocumentIntelligenceOcrSdkInput:
    return AzureDocumentIntelligenceOcrSdkInput.model_validate(
        {"model": model, "document": document, **(optional_params or {})}
    )


def azure_document_intelligence_input_strategy() -> SearchStrategy[AzureDocumentIntelligenceOcrSdkInput]:
    document: Final = pdf_document()
    pages: Final = st.one_of(
        st.sampled_from(((0,), (2, 0, 0, 1))).map(list),
        st.just(["1", "2-4"]),
        st.just("1-4, 5"),
    ).map(lambda value: {"pages": value})
    features: Final = st.one_of(
        st.sampled_from(
            (
                ("languages",),
                ("ocrHighResolution",),
                ("barcodes",),
                ("formulas",),
                ("styleFont",),
                ("keyValuePairs",),
            )
        ).map(list),
        st.just("languages, styleFont"),
    ).map(lambda value: {"features": value})
    combined_query: Final = st.just({"pages": (0, 1), "features": ("languages", "styleFont")})
    return st.one_of(
        st.sampled_from(AZURE_DOCUMENT_INTELLIGENCE_RECORDING_MODELS).map(
            lambda model: _document_intelligence_input(model, document)
        ),
        st.just(
            _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL,
                image_document("invoice 123", 24),
            )
        ),
        pages.map(
            lambda optional_params: _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL, document, optional_params
            )
        ),
        features.map(
            lambda optional_params: _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL, document, optional_params
            )
        ),
        combined_query.map(
            lambda optional_params: _document_intelligence_input(
                _AZURE_DOCUMENT_INTELLIGENCE_CANONICAL_MODEL, document, optional_params
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
    environ: Mapping[str, str], client: OcrFixtureClient, inline_image_data_uri: str
) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("AZURE_AI_API_KEY")
    base_url: Final = environ.get("AZURE_AI_API_BASE")
    if not api_key or not base_url:
        return ()
    return (
        OcrRecordingTarget(
            name="azure-mistral",
            upstream=UpstreamEndpoint(base_url=base_url.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                azure_mistral_input_strategy(inline_image_data_uri),
            ),
            invocation=invoke_with_api_key(client, api_key),
            required_inputs=AZURE_MISTRAL_PROVIDER_REJECTED_INPUTS,
        ),
    )


def azure_document_intelligence_recording_targets(
    environ: Mapping[str, str], client: OcrFixtureClient
) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    base_url: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not api_key or not base_url:
        return ()
    return (
        OcrRecordingTarget(
            name="azure-document-intelligence",
            upstream=UpstreamEndpoint(base_url=base_url.rstrip("/")),
            strategy=cast(SearchStrategy[OcrSdkInputBase], azure_document_intelligence_input_strategy()),
            invocation=invoke_with_api_key(client, api_key),
            required_inputs=AZURE_DOCUMENT_INTELLIGENCE_PROVIDER_REJECTED_INPUTS,
        ),
    )
