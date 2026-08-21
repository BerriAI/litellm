"""
Regression tests for Azure Document Intelligence api_base resolution in OCR.

`azure_ai` exposes two OCR services on one provider; the `doc-intelligence`
sub-route must resolve to `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, not to the
generic `AZURE_AI_API_BASE` fallback that `get_llm_provider` injects. These tests
pin that routing and guard the backwards-compatibility contract that an explicitly
supplied api_base is always honoured.
"""

from litellm.llms.azure_ai.ocr.common_utils import (
    is_azure_document_intelligence_model,
)
from litellm.ocr.main import _prepare_ocr_request, _rust_bridge_api_base

_DOC = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
_DOC_INTELLIGENCE_ENDPOINT = "https://di.cognitiveservices.azure.com"
_AZURE_AI_API_BASE = "https://generic-azure-ai.example.com"


class _FakeLogging:
    def update_from_kwargs(self, **kwargs: object) -> None:
        return None


def _resolve_secret(name: str) -> str | None:
    return {
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": _DOC_INTELLIGENCE_ENDPOINT,
        "AZURE_AI_API_BASE": _AZURE_AI_API_BASE,
    }.get(name)


def _prepare(model: str, api_base: str | None):
    return _prepare_ocr_request(
        model=model,
        document=dict(_DOC),
        api_key="test-key",
        api_base=api_base,
        timeout=None,
        custom_llm_provider=None,
        extra_headers=None,
        kwargs={"litellm_logging_obj": _FakeLogging()},
    )


class TestIsAzureDocumentIntelligenceModel:
    def test_matches_doc_intelligence_route(self):
        assert is_azure_document_intelligence_model("doc-intelligence/prebuilt-layout")

    def test_matches_documentintelligence_and_is_case_insensitive(self):
        assert is_azure_document_intelligence_model("azure_ai/DocumentIntelligence/x")

    def test_does_not_match_mistral_route(self):
        assert not is_azure_document_intelligence_model("mistral-document-ai-2505")


class TestDocIntelligenceApiBaseResolution:
    def test_generic_azure_ai_base_does_not_hijack_doc_intelligence(self, monkeypatch):
        """Without an explicit api_base, the AZURE_AI_API_BASE fallback must not
        overwrite the endpoint, so it resolves to the Document Intelligence one."""
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)

        prepared = _prepare("azure_ai/doc-intelligence/prebuilt-layout", None)

        assert prepared.api_base is None
        assert _rust_bridge_api_base(prepared, _resolve_secret) == _DOC_INTELLIGENCE_ENDPOINT

    def test_explicit_api_base_is_honoured_for_doc_intelligence(self, monkeypatch):
        """A caller-supplied api_base must always win, even for doc-intelligence."""
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        custom = "https://my-di.cognitiveservices.azure.com"
        prepared = _prepare("azure_ai/doc-intelligence/prebuilt-layout", custom)

        assert prepared.api_base == custom
        assert _rust_bridge_api_base(prepared, _resolve_secret) == custom

    def test_generic_azure_ai_base_still_applies_to_mistral_ocr(self, monkeypatch):
        """Non doc-intelligence azure_ai models keep using AZURE_AI_API_BASE."""
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        prepared = _prepare("azure_ai/mistral-document-ai-2505", None)

        assert prepared.api_base == _AZURE_AI_API_BASE
