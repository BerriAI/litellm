"""
Test OCR functionality with Azure Document Intelligence API.

Azure Document Intelligence provides advanced document analysis capabilities
using the v4.0 (2024-11-30) API.
"""

import os
from typing import Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from base_ocr_unit_tests import BaseOCRTest
from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
    AzureDocumentIntelligenceOCRConfig,
)

def _make_config() -> AzureDocumentIntelligenceOCRConfig:
    return AzureDocumentIntelligenceOCRConfig()


def _make_202_response(
    operation_url: str = "https://example.com/operations/123",
    *,
    auth_header: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> MagicMock:
    """Build a mock 202 response whose .request.headers reflects auth."""
    req_headers: dict = {}
    if auth_header is not None:
        req_headers["Ocp-Apim-Subscription-Key"] = auth_header
    if bearer_token is not None:
        req_headers["Authorization"] = bearer_token

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 202
    mock_response.headers = {"Operation-Location": operation_url}
    mock_response.request = MagicMock()
    mock_response.request.headers = req_headers
    return mock_response


def _make_succeeded_response() -> MagicMock:
    """Build a mock 200 response with a minimal succeeded analyzeResult."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "succeeded",
        "analyzeResult": {
            "pages": [
                {
                    "pageNumber": 1,
                    "width": 8.5,
                    "height": 11,
                    "unit": "inch",
                    "lines": [{"content": "Hello world"}],
                }
            ]
        },
    }
    return mock_response

class TestValidateEnvironmentAuth:
    """Tests for the validate_environment auth logic."""

    def test_should_use_api_key_header_when_api_key_is_provided(self):
        """When api_key is given, Ocp-Apim-Subscription-Key must be set."""
        config = _make_config()
        headers = config.validate_environment(
            headers={},
            model="prebuilt-layout",
            api_key="my-key",
            api_base="https://my.cognitiveservices.azure.com",
        )

        assert headers["Ocp-Apim-Subscription-Key"] == "my-key"
        assert "Authorization" not in headers

    def test_should_read_api_key_from_env_when_not_passed(self, monkeypatch):
        """API key is read from AZURE_DOCUMENT_INTELLIGENCE_API_KEY when not passed."""
        monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", "env-key")
        monkeypatch.setenv(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "https://my.cognitiveservices.azure.com",
        )
        config = _make_config()
        headers = config.validate_environment(headers={}, model="prebuilt-layout")

        assert headers["Ocp-Apim-Subscription-Key"] == "env-key"
        assert "Authorization" not in headers

    def test_should_use_bearer_token_when_no_api_key(self, monkeypatch):
        """When no API key is available, get_azure_ad_token should be called and the
        Authorization header should be set with the returned bearer token."""
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", raising=False)

        config = _make_config()
        # get_azure_ad_token is a local import inside validate_environment; patch at source
        with patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            return_value="fake-bearer-token",
        ):
            headers = config.validate_environment(
                headers={},
                model="prebuilt-layout",
                api_key=None,
                api_base="https://my.cognitiveservices.azure.com",
                litellm_params={
                    "tenant_id": "t",
                    "client_id": "c",
                    "client_secret": "s",
                },
            )

        assert headers["Authorization"] == "Bearer fake-bearer-token"
        assert "Ocp-Apim-Subscription-Key" not in headers

    def test_should_forward_litellm_params_to_get_azure_ad_token(self, monkeypatch):
        """litellm_params passed to validate_environment must be forwarded to
        get_azure_ad_token as a GenericLiteLLMParams."""
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", raising=False)

        config = _make_config()
        captured = {}

        def fake_get_token(params):
            captured["params"] = params
            return "tok"

        # get_azure_ad_token is a local import inside validate_environment; patch at source
        with patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            side_effect=fake_get_token,
        ):
            config.validate_environment(
                headers={},
                model="prebuilt-layout",
                api_key=None,
                api_base="https://my.cognitiveservices.azure.com",
                litellm_params={"tenant_id": "tenant-x"},
            )

        assert captured["params"].get("tenant_id") == "tenant-x"

    def test_should_raise_when_neither_api_key_nor_ad_token_available(self, monkeypatch):
        """When both api_key and get_azure_ad_token return None, a ValueError must be
        raised with a helpful message."""
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", raising=False)

        config = _make_config()
        # get_azure_ad_token is a local import inside validate_environment; patch at source
        with patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="Missing Azure Document Intelligence credentials"):
                config.validate_environment(
                    headers={},
                    model="prebuilt-layout",
                    api_key=None,
                    api_base="https://my.cognitiveservices.azure.com",
                )

    def test_should_raise_when_api_base_is_missing(self, monkeypatch):
        """Missing endpoint should raise even if api_key is provided."""
        monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)

        config = _make_config()
        with pytest.raises(ValueError, match="Missing Azure Document Intelligence Endpoint"):
            config.validate_environment(
                headers={},
                model="prebuilt-layout",
                api_key="some-key",
                api_base=None,
            )

    def test_should_preserve_caller_supplied_headers(self):
        """Extra headers supplied by the caller must survive in the merged output."""
        config = _make_config()
        headers = config.validate_environment(
            headers={"X-Custom-Header": "value"},
            model="prebuilt-layout",
            api_key="my-key",
            api_base="https://my.cognitiveservices.azure.com",
        )

        assert headers["X-Custom-Header"] == "value"

class TestTransformOcrResponsePollHeaders:
    """Verify the correct auth header is forwarded when polling (sync path)."""

    def _run_with_mock_poll(self, raw_202_response, captured: dict):
        config = _make_config()
        succeeded = _make_succeeded_response()

        def fake_poll(operation_url, headers, timeout_secs):
            captured["headers"] = dict(headers)
            return succeeded

        with patch.object(config, "_poll_operation_sync", side_effect=fake_poll):
            config.transform_ocr_response(
                model="prebuilt-layout",
                raw_response=raw_202_response,
                logging_obj=MagicMock(),
            )

    def test_should_forward_subscription_key_when_present(self):
        raw = _make_202_response(auth_header="my-api-key")
        captured: dict = {}
        self._run_with_mock_poll(raw, captured)

        assert captured["headers"].get("Ocp-Apim-Subscription-Key") == "my-api-key"
        assert "Authorization" not in captured["headers"]

    def test_should_forward_bearer_token_when_present(self):
        raw = _make_202_response(bearer_token="Bearer fake-token")
        captured: dict = {}
        self._run_with_mock_poll(raw, captured)

        assert captured["headers"].get("Authorization") == "Bearer fake-token"
        assert "Ocp-Apim-Subscription-Key" not in captured["headers"]

    def test_should_raise_when_neither_header_present(self):
        raw = _make_202_response()
        config = _make_config()

        with pytest.raises(ValueError, match="no authentication header"):
            config.transform_ocr_response(
                model="prebuilt-layout",
                raw_response=raw,
                logging_obj=MagicMock(),
            )

    def test_should_prefer_subscription_key_over_bearer_when_both_present(self):
        """Ocp-Apim-Subscription-Key takes priority (if before elif in production code)."""
        raw = _make_202_response(auth_header="api-key-value", bearer_token="Bearer bearer-value")
        captured: dict = {}
        self._run_with_mock_poll(raw, captured)

        assert captured["headers"].get("Ocp-Apim-Subscription-Key") == "api-key-value"
        assert "Authorization" not in captured["headers"]

class TestAsyncTransformOcrResponsePollHeaders:
    """Verify the correct auth header is forwarded when polling (async path)."""

    async def _run_with_mock_poll(self, raw_202_response, captured: dict):
        config = _make_config()
        succeeded = _make_succeeded_response()

        async def fake_poll(operation_url, headers, timeout_secs):
            captured["headers"] = dict(headers)
            return succeeded

        with patch.object(config, "_poll_operation_async", side_effect=fake_poll):
            await config.async_transform_ocr_response(
                model="prebuilt-layout",
                raw_response=raw_202_response,
                logging_obj=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_should_forward_subscription_key_when_present(self):
        raw = _make_202_response(auth_header="my-api-key")
        captured: dict = {}
        await self._run_with_mock_poll(raw, captured)

        assert captured["headers"].get("Ocp-Apim-Subscription-Key") == "my-api-key"
        assert "Authorization" not in captured["headers"]

    @pytest.mark.asyncio
    async def test_should_forward_bearer_token_when_present(self):
        raw = _make_202_response(bearer_token="Bearer fake-token")
        captured: dict = {}
        await self._run_with_mock_poll(raw, captured)

        assert captured["headers"].get("Authorization") == "Bearer fake-token"
        assert "Ocp-Apim-Subscription-Key" not in captured["headers"]

    @pytest.mark.asyncio
    async def test_should_raise_when_neither_header_present(self):
        raw = _make_202_response()
        config = _make_config()

        with pytest.raises(ValueError, match="no authentication header"):
            await config.async_transform_ocr_response(
                model="prebuilt-layout",
                raw_response=raw,
                logging_obj=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_should_prefer_subscription_key_over_bearer_when_both_present(self):
        """Ocp-Apim-Subscription-Key takes priority (if before elif in production code)."""
        raw = _make_202_response(auth_header="api-key-value", bearer_token="Bearer bearer-value")
        captured: dict = {}
        await self._run_with_mock_poll(raw, captured)

        assert captured["headers"].get("Ocp-Apim-Subscription-Key") == "api-key-value"
        assert "Authorization" not in captured["headers"]

class TestAzureDocumentIntelligenceOCR(BaseOCRTest):
    """
    Test class for Azure Document Intelligence OCR functionality.
    
    Inherits from BaseOCRTest and provides Azure Document Intelligence-specific configuration.
    
    Tests the azure_ai/doc-intelligence/<model> provider route.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Azure Document Intelligence.
        
        Uses prebuilt-layout model which is closest to Mistral OCR format.
        """
        # Check for required environment variables
        api_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if not api_key or not endpoint:
            pytest.skip(
                "AZURE_DOCUMENT_INTELLIGENCE_API_KEY and AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "environment variables are required for Azure Document Intelligence tests"
            )

        return {
            "model": "azure_ai/doc-intelligence/prebuilt-layout",
            "api_key": api_key,
            "api_base": endpoint,
        }

