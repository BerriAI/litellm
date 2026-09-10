from inspect import isawaitable
from typing import Final

import pytest

from litellm.llms.azure_ai.ocr.common_utils import (
    is_azure_document_intelligence_model,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.ocr.main import _prepare_ocr_request, _PreparedOCRRequest, _rust_bridge_api_base

_DOC = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
_DOC_INTELLIGENCE_ENDPOINT = "https://di.cognitiveservices.azure.com"
_AZURE_AI_API_BASE = "https://generic-azure-ai.example.com"
_DOC_INTELLIGENCE_API_KEY = "document-intelligence-key"
_AZURE_AI_API_KEY = "generic-azure-ai-key"


class _FakeLogging:
    def update_from_kwargs(self, **kwargs: object) -> None:
        return None

    def pre_call(self, **kwargs: object) -> None:
        return None


def _resolve_secret(name: str) -> str | None:
    return {
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": _DOC_INTELLIGENCE_ENDPOINT,
        "AZURE_AI_API_BASE": _AZURE_AI_API_BASE,
    }.get(name)


def _prepare(model: str, api_base: str | None = None, *, api_key: str | None = None) -> _PreparedOCRRequest:
    return _prepare_ocr_request(
        model=model,
        document=dict(_DOC),
        api_key=api_key,
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
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)

        prepared = _prepare("azure_ai/doc-intelligence/prebuilt-layout", None)

        assert prepared.api_base is None
        assert _rust_bridge_api_base(prepared, _resolve_secret) == _DOC_INTELLIGENCE_ENDPOINT

    def test_explicit_api_base_is_honoured_for_doc_intelligence(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        custom = "https://my-di.cognitiveservices.azure.com"
        prepared = _prepare("azure_ai/doc-intelligence/prebuilt-layout", custom)

        assert prepared.api_base == custom
        assert _rust_bridge_api_base(prepared, _resolve_secret) == custom

    def test_generic_azure_ai_base_still_applies_to_mistral_ocr(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

        prepared = _prepare("azure_ai/mistral-document-ai-2505", None)

        assert prepared.api_base == _AZURE_AI_API_BASE


class TestDocIntelligenceApiKeyResolution:
    def test_generic_azure_credentials_are_not_forwarded_to_doc_intelligence(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_KEY", _AZURE_AI_API_KEY)
        monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)

        prepared = _prepare("azure_ai/doc-intelligence/prebuilt-layout", None, api_key=None)

        assert prepared.api_key is None
        assert prepared.api_base is None

    def test_generic_azure_ai_key_still_applies_to_mistral_ocr(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_KEY", _AZURE_AI_API_KEY)

        prepared = _prepare("azure_ai/mistral-document-ai-2505", None, api_key=None)

        assert prepared.api_key == _AZURE_AI_API_KEY


@pytest.mark.parametrize("use_async", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(
    "api_key, api_base, expected_key, expected_endpoint",
    (
        pytest.param(None, None, _DOC_INTELLIGENCE_API_KEY, _DOC_INTELLIGENCE_ENDPOINT, id="environment"),
        pytest.param(
            "explicit-key",
            "https://explicit.example.com",
            "explicit-key",
            "https://explicit.example.com",
            id="explicit",
        ),
    ),
)
@pytest.mark.asyncio
async def test_document_intelligence_request_uses_its_own_credentials(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    api_key: str | None,
    api_base: str | None,
    expected_key: str,
    expected_endpoint: str,
) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", _AZURE_AI_API_KEY)
    monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", _DOC_INTELLIGENCE_API_KEY)
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", _DOC_INTELLIGENCE_ENDPOINT)
    prepared: Final = _prepare("azure_ai/doc-intelligence/prebuilt-layout", api_base, api_key=api_key)
    handler: Final = BaseLLMHTTPHandler()
    prepare_request: Final = handler._async_prepare_ocr_request if use_async else handler._prepare_ocr_request
    result: Final = prepare_request(
        model=prepared.model,
        document=prepared.document,
        optional_params=prepared.optional_params,
        logging_obj=_FakeLogging(),
        api_key=prepared.api_key,
        api_base=prepared.api_base,
        headers=None,
        provider_config=prepared.provider_config,
        litellm_params=prepared.litellm_params,
    )

    headers, url, _, _ = await result if isawaitable(result) else result

    assert headers["Ocp-Apim-Subscription-Key"] == expected_key
    assert url.startswith(f"{expected_endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze?")


def test_explicit_secret_references_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", _AZURE_AI_API_KEY)
    monkeypatch.setenv("AZURE_AI_API_BASE", _AZURE_AI_API_BASE)

    prepared: Final = _prepare(
        "azure_ai/doc-intelligence/prebuilt-layout",
        "os.environ/AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        api_key="os.environ/AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    )

    assert prepared.api_key == "os.environ/AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
    assert prepared.api_base == "os.environ/AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
