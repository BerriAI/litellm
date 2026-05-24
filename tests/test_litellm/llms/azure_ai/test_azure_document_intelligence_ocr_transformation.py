import pytest

from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
    AzureDocumentIntelligenceOCRConfig,
)


def test_should_encode_azure_document_intelligence_model_id():
    config = AzureDocumentIntelligenceOCRConfig()

    url = config.get_complete_url(
        api_base="https://example.cognitiveservices.azure.com",
        model="prebuilt-layout?x=1#frag",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout%3Fx%3D1%23frag:analyze?api-version=2024-11-30"
    )


def test_should_reject_dot_segment_azure_document_intelligence_model_id():
    config = AzureDocumentIntelligenceOCRConfig()

    with pytest.raises(ValueError, match="model_id cannot be a dot path segment"):
        config.get_complete_url(
            api_base="https://example.cognitiveservices.azure.com",
            model="azure_ai/doc-intelligence/..",
            optional_params={},
            litellm_params={},
        )
