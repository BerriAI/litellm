"""Azure Document Intelligence api_base resolution regression tests."""

from litellm.ocr.main import _resolve_ocr_call_context

_DOC = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
_AZURE_AI_API_BASE = "https://generic-azure-ai.example.com"


class _FakeLogging:
    def update_from_kwargs(self, **kwargs: object) -> None:
        return None


def _resolved_api_base(model: str, api_base: str | None) -> str | None:
    result = _resolve_ocr_call_context(
        model=model,
        document=dict(_DOC),
        api_key="test-key",
        api_base=api_base,
        timeout=None,
        custom_llm_provider=None,
        extra_headers=None,
        kwargs={"litellm_logging_obj": _FakeLogging()},
    )
    return result[3]


class TestDocIntelligenceApiBaseResolution:
    def test_generic_azure_ai_base_does_not_hijack_doc_intelligence(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        assert (
            _resolved_api_base("azure_ai/doc-intelligence/prebuilt-layout", None)
            is None
        )

    def test_explicit_api_base_is_honoured_for_doc_intelligence(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        custom = "https://my-di.cognitiveservices.azure.com"

        assert (
            _resolved_api_base("azure_ai/doc-intelligence/prebuilt-layout", custom)
            == custom
        )

    def test_generic_azure_ai_base_still_applies_to_mistral_ocr(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        assert (
            _resolved_api_base("azure_ai/mistral-document-ai-2505", None)
            == _AZURE_AI_API_BASE
        )
