from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from tests.route_parity.fixture_recorder import ProviderSpec
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrFixtureTarget,
    invoke_with_api_key,
    pdf_document,
    public_document_strategy,
)
from tests.test_litellm.ocr.fixtures.mistral import (
    MISTRAL_MODEL,
    mistral_input_strategy,
    required_mistral_inputs,
)
from tests.test_litellm.ocr.fixtures.models import (
    AzureDocumentIntelligenceOcrSdkInput,
    AzureMistralOcrSdkInput,
    MistralOcrSdkInput,
    OcrSdkInputBase,
)


def _as_azure_mistral(case_input: MistralOcrSdkInput, model: str) -> AzureMistralOcrSdkInput:
    values: Final = case_input.model_dump(mode="python", exclude={"boundary", "model", "custom_llm_provider"})
    return AzureMistralOcrSdkInput.model_validate({**values, "model": model})


def _required_document_intelligence_inputs() -> tuple[AzureDocumentIntelligenceOcrSdkInput, ...]:
    document: Final = pdf_document()
    model: Final = "azure_ai/doc-intelligence/prebuilt-layout"
    return (
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document),
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document, pages=[0, 1]),
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document, features=["languages"]),
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document, req_format="litellm"),
    )


@st.composite
def azure_document_intelligence_input_strategy(draw: DrawFn) -> AzureDocumentIntelligenceOcrSdkInput:
    optional_params: Final = draw(
        st.fixed_dictionaries(
            {},
            optional={
                "pages": st.sampled_from(([0], [0, 1], "1-2")),
                "features": st.sampled_from((["languages"], ["keyValuePairs"], "languages,keyValuePairs")),
            },
        )
    )
    return AzureDocumentIntelligenceOcrSdkInput.model_validate(
        {
            "model": draw(
                st.sampled_from(
                    (
                        "azure_ai/doc-intelligence/prebuilt-read",
                        "azure_ai/doc-intelligence/prebuilt-layout",
                        "azure_ai/doc-intelligence/prebuilt-document",
                    )
                )
            ),
            "document": draw(public_document_strategy()),
            **optional_params,
        }
    )


class AzureMistralFixtureSource:
    def targets(self, environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrFixtureTarget, ...]:
        api_key: Final = environ.get("AZURE_AI_API_KEY")
        upstream_base: Final = environ.get("AZURE_AI_API_BASE")
        configured_model: Final = environ.get("AZURE_AI_OCR_MODEL")
        if not api_key or not upstream_base or not configured_model:
            return ()
        model: Final = configured_model if configured_model.startswith("azure_ai/") else f"azure_ai/{configured_model}"
        return (
            OcrFixtureTarget(
                name="azure-mistral",
                provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
                strategy=cast(
                    SearchStrategy[OcrSdkInputBase],
                    mistral_input_strategy(MISTRAL_MODEL).map(lambda case_input: _as_azure_mistral(case_input, model)),
                ),
                invocation=invoke_with_api_key(client, api_key),
                required_inputs=cast(
                    tuple[OcrSdkInputBase, ...],
                    tuple(
                        _as_azure_mistral(case_input, model) for case_input in required_mistral_inputs(MISTRAL_MODEL)
                    ),
                ),
            ),
        )


class AzureDocumentIntelligenceFixtureSource:
    def targets(self, environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrFixtureTarget, ...]:
        api_key: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        upstream_base: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        if not api_key or not upstream_base:
            return ()
        return (
            OcrFixtureTarget(
                name="azure-document-intelligence",
                provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
                strategy=cast(SearchStrategy[OcrSdkInputBase], azure_document_intelligence_input_strategy()),
                invocation=invoke_with_api_key(client, api_key),
                required_inputs=cast(tuple[OcrSdkInputBase, ...], _required_document_intelligence_inputs()),
            ),
        )
