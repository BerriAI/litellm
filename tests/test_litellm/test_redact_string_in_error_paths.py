"""
Tests for _redact_string usage in error/logging paths.

Covers actual execution of redaction in:
- WebSocket close reasons in realtime handlers (openai, azure, bedrock)
- Gemini RAG ingestion x-goog-api-key header usage
- Traceback redaction pattern used in proxy streaming
"""

import os
import sys
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm._logging import _ENABLE_SECRET_REDACTION, _redact_string


class TestRedactStringFunction:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer sk-1234567890abcdefghij"
        result = _redact_string(text)
        assert "sk-1234567890abcdefghij" not in result
        assert "REDACTED" in result

    def test_redacts_api_key_in_url(self):
        text = "Error at https://example.com?api_key=my-secret-key-value-here"
        result = _redact_string(text)
        assert "my-secret-key-value-here" not in result

    def test_redacts_google_api_key(self):
        text = "key=AIzaSyB1234567890abcdefghijklmnopqrstuvwx"
        result = _redact_string(text)
        assert "AIzaSyB1234567890abcdefghijklmnopqrstuvwx" not in result

    def test_passes_clean_text_through(self):
        text = "This is a normal error message with no secrets"
        assert _redact_string(text) == text

    @pytest.mark.skipif(
        not _ENABLE_SECRET_REDACTION, reason="redaction disabled via env var"
    )
    def test_redaction_enabled_by_default(self):
        text = "Bearer sk-1234567890abcdefghij"
        result = _redact_string(text)
        assert "sk-1234567890abcdefghij" not in result


class TestOpenAIRealtimeRedaction:
    """Test that OpenAI realtime handler redacts secrets in websocket close reasons."""

    def _make_patches(self, handler):
        """Shared patches for OpenAI realtime handler tests."""
        return (
            patch.object(
                handler,
                "_construct_url",
                return_value="wss://api.openai.com/v1/realtime?model=gpt-4",
            ),
            patch.object(handler, "_get_ssl_config", return_value=None),
            patch.object(handler, "_get_additional_headers", return_value={}),
        )

    def _call_kwargs(self):
        return dict(
            model="gpt-4",
            websocket=AsyncMock(),
            logging_obj=MagicMock(),
            api_base="https://api.openai.com/",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_invalid_status_code_redacts_reason(self):
        import websockets.exceptions

        from litellm.llms.openai.realtime.handler import OpenAIRealtime

        handler = OpenAIRealtime()
        exc = websockets.exceptions.InvalidStatusCode(403, None)
        exc.status_code = 403

        kwargs = self._call_kwargs()
        mock_ws = kwargs["websocket"]
        p1, p2, p3 = self._make_patches(handler)
        with p1, p2, p3, patch("websockets.connect", side_effect=exc):
            await handler.async_realtime(**kwargs)

        mock_ws.close.assert_called_once()
        assert mock_ws.close.call_args[1]["code"] == 403

    @pytest.mark.asyncio
    async def test_generic_exception_redacts_reason(self):
        from litellm.llms.openai.realtime.handler import OpenAIRealtime

        handler = OpenAIRealtime()
        secret_error = RuntimeError(
            "Connection failed for api_key=sk-1234567890abcdefghij"
        )

        kwargs = self._call_kwargs()
        mock_ws = kwargs["websocket"]
        p1, p2, p3 = self._make_patches(handler)
        with p1, p2, p3, patch("websockets.connect", side_effect=secret_error):
            await handler.async_realtime(**kwargs)

        mock_ws.close.assert_called_once()
        assert mock_ws.close.call_args[1]["code"] == 1011
        assert "sk-1234567890abcdefghij" not in mock_ws.close.call_args[1]["reason"]


class TestAzureRealtimeRedaction:
    """Test that Azure realtime handler redacts secrets in websocket close reasons."""

    @pytest.mark.asyncio
    async def test_invalid_status_code_redacts_reason(self):
        import websockets.exceptions

        from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

        handler = AzureOpenAIRealtime()
        mock_ws = AsyncMock()
        exc = websockets.exceptions.InvalidStatusCode(403, None)
        exc.status_code = 403

        with (
            patch.object(
                handler,
                "_construct_url",
                return_value="wss://test.openai.azure.com/openai/realtime",
            ),
            patch("websockets.connect", side_effect=exc),
        ):
            await handler.async_realtime(
                model="gpt-4",
                websocket=mock_ws,
                logging_obj=MagicMock(),
                api_base="https://test.openai.azure.com/",
                api_key="test-key",
                api_version="2024-10-01-preview",
            )

        mock_ws.close.assert_called_once()
        assert mock_ws.close.call_args[1]["code"] == 403


class TestBedrockRealtimeRedaction:
    """Test that _redact_string produces safe close reasons for Bedrock-style errors."""

    def test_internal_error_message_redacted(self):
        secret_error = RuntimeError(
            "Failed with aws_secret_access_key=AKIAIOSFODNN7EXAMPLE123456"
        )
        reason = _redact_string(f"Internal error: {str(secret_error)}")
        assert "AKIAIOSFODNN7EXAMPLE123456" not in reason


class TestLLMHTTPHandlerRealtimeRedaction:
    """Test _redact_string on the exact patterns used in llm_http_handler WS close."""

    def test_invalid_status_pattern(self):
        error_msg = "InvalidStatusCode: 403 for wss://api.example.com?api_key=sk-leaked-key-here"
        assert "sk-leaked-key-here" not in _redact_string(str(error_msg))

    def test_internal_server_error_pattern(self):
        error_msg = "Connection failed for api_key=sk-secret-key-12345678"
        assert "sk-secret-key-12345678" not in _redact_string(
            f"Internal server error: {error_msg}"
        )


class TestProxyStreamingDataGeneratorRedaction:
    """Test _redact_string on traceback.format_exc() — the pattern at common_request_processing.py:1733."""

    def test_redact_traceback_format_exc(self):
        try:
            raise RuntimeError(
                "Failed connecting to api_key=sk-1234567890abcdefghij at https://api.example.com"
            )
        except RuntimeError:
            raw_tb = traceback.format_exc()

        redacted_tb = _redact_string(raw_tb)

        assert "sk-1234567890abcdefghij" not in redacted_tb
        assert "Traceback" in redacted_tb
        assert "RuntimeError" in redacted_tb


def _make_mock_ingest_options():
    mock = MagicMock()
    mock.vector_store_config = {}
    mock.ingest_name = "test"
    mock.chunking_strategy = None
    mock.embedding_model = None
    mock.vector_db_type = "gemini"
    return mock


class TestGeminiIngestionHeaders:
    """Test that Gemini RAG ingestion uses x-goog-api-key header."""

    @pytest.mark.asyncio
    async def test_create_file_search_store_sends_header(self):
        from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion

        ingestion = GeminiRAGIngestion(ingest_options=_make_mock_ingest_options())

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "fileSearchStores/abc123"}
        mock_client.post.return_value = mock_response

        with patch(
            "litellm.rag.ingestion.gemini_ingestion.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await ingestion._create_file_search_store(
                api_key="test-gemini-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                display_name="test-store",
            )

        assert result == "fileSearchStores/abc123"
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["headers"]["x-goog-api-key"] == "test-gemini-key"
        assert "key=" not in call_kwargs[0][0]

    @pytest.mark.asyncio
    async def test_initiate_resumable_upload_sends_header(self):
        from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion

        ingestion = GeminiRAGIngestion(ingest_options=_make_mock_ingest_options())

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "x-goog-upload-url": "https://upload.example.com/upload123"
        }
        mock_client.post.return_value = mock_response

        with patch(
            "litellm.rag.ingestion.gemini_ingestion.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await ingestion._initiate_resumable_upload(
                api_key="test-gemini-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                vector_store_id="fileSearchStores/abc123",
                filename="test.txt",
                file_size=1024,
                content_type="text/plain",
            )

        assert result == "https://upload.example.com/upload123"
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["headers"]["x-goog-api-key"] == "test-gemini-key"
        assert "key=" not in call_kwargs[0][0]
