"""
Tests for redact_string usage in error/logging paths.

Covers actual execution of redaction in:
- WebSocket close reasons in realtime handlers (openai, bedrock)
- Gemini RAG ingestion x-goog-api-key header usage
- Traceback redaction pattern used in proxy streaming
- Router fallback-failure traceback redaction
"""

import logging
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm._logging import _ENABLE_SECRET_REDACTION, redact_string


class TestRedactStringFunction:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer sk-9876567890abcdefghij"
        result = redact_string(text)
        assert "sk-9876567890abcdefghij" not in result
        assert "REDACTED" in result

    def test_redacts_api_key_in_url(self):
        text = "Error at https://example.com?api_key=my-secret-key-value-here"
        result = redact_string(text)
        assert "my-secret-key-value-here" not in result

    def test_redacts_google_api_key(self):
        text = "key=AIzaSyB1234567890abcdefghijklmnopqrstuvwx"
        result = redact_string(text)
        assert "AIzaSyB1234567890abcdefghijklmnopqrstuvwx" not in result

    def test_passes_clean_text_through(self):
        text = "This is a normal error message with no secrets"
        assert redact_string(text) == text

    @pytest.mark.skipif(not _ENABLE_SECRET_REDACTION, reason="redaction disabled via env var")
    def test_redaction_enabled_by_default(self):
        text = "Bearer sk-9876567890abcdefghij"
        result = redact_string(text)
        assert "sk-9876567890abcdefghij" not in result


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
    async def test_generic_exception_redacts_reason(self):
        from litellm.llms.openai.realtime.handler import OpenAIRealtime

        handler = OpenAIRealtime()
        secret_error = RuntimeError("Connection failed for api_key=sk-9876567890abcdefghij")

        kwargs = self._call_kwargs()
        mock_ws = kwargs["websocket"]
        p1, p2, p3 = self._make_patches(handler)
        with p1, p2, p3, patch("websockets.connect", side_effect=secret_error):
            await handler.async_realtime(**kwargs)

        mock_ws.close.assert_called_once()
        assert mock_ws.close.call_args[1]["code"] == 1011
        assert "sk-9876567890abcdefghij" not in mock_ws.close.call_args[1]["reason"]


class TestBedrockRealtimeRedaction:
    """Test that redact_string produces safe close reasons for Bedrock-style errors."""

    def test_internal_error_message_redacted(self):
        secret_error = RuntimeError("Failed with aws_secret_access_key=AKIAIOSFODNN7EXAMPLE123456")
        reason = redact_string(f"Internal error: {str(secret_error)}")
        assert "AKIAIOSFODNN7EXAMPLE123456" not in reason


class TestLLMHTTPHandlerRealtimeRedaction:
    """Test redact_string on the exact patterns used in llm_http_handler WS close."""

    def test_invalid_status_pattern(self):
        error_msg = "InvalidStatusCode: 403 for wss://api.example.com?api_key=sk-leaked-key-here"
        assert "sk-leaked-key-here" not in redact_string(str(error_msg))

    def test_internal_server_error_pattern(self):
        error_msg = "Connection failed for api_key=sk-secret-key-12345678"
        assert "sk-secret-key-12345678" not in redact_string(f"Internal server error: {error_msg}")


class TestProxyStreamingDataGeneratorRedaction:
    def test_redact_traceback_format_exc(self):
        try:
            raise RuntimeError("Failed connecting to api_key=sk-9876567890abcdefghij at https://api.example.com")
        except RuntimeError:
            raw_tb = traceback.format_exc()

        redacted_tb = redact_string(raw_tb)

        assert "sk-9876567890abcdefghij" not in redacted_tb
        assert "Traceback" in redacted_tb
        assert "RuntimeError" in redacted_tb


class TestRouterFallbackFailureTracebackRedaction:
    """Test the router's fallback-failure logs. Both call sites must redact the
    traceback at the call site with redact_string() rather than hand a live
    exception to exc_info=True. SecretRedactionFilter rewrites record.exc_text,
    but record.exc_info stays an exception object no filter can rewrite, so any
    handler that renders exc_info itself (Datadog and OTel log bridges do) would
    receive the unredacted secret."""

    @pytest.mark.asyncio
    async def test_fallback_failure_does_not_leak_secret_via_exc_info(self, caplog):
        """Both the primary group and its fallback fail with the secret in the error, so the
        router logs the fallback failure while that exception is in flight."""
        import litellm

        secret = "sk-testsecretvalue1234567890abcdef"
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "fake-key",
                        "mock_response": Exception(f"primary deployment failed api_key={secret}"),
                    },
                },
                {
                    "model_name": "claude-3-haiku",
                    "litellm_params": {
                        "model": "anthropic/claude-3-haiku-20240307",
                        "api_key": "fake-key",
                        "mock_response": Exception(f"boom api_key={secret}"),
                    },
                },
            ],
            fallbacks=[{"gpt-3.5-turbo": ["claude-3-haiku"]}],
        )

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
            with pytest.raises(litellm.InternalServerError):
                await router.acompletion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hi"}])

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert debug_records, "expected the entry-point debug log, which carries the active traceback"

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("Error occurred while trying to do fallbacks" in r.getMessage() for r in error_records), (
            "expected the fallback-failure log, not an unrelated error"
        )

        for record in caplog.records:
            assert secret not in record.getMessage()
            assert secret not in (record.exc_text or "")
            rendered_exc_info = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else ""
            assert secret not in rendered_exc_info, (
                f"{record.levelname} record passed a live exception to exc_info; "
                "no logging filter can redact record.exc_info"
            )


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
        mock_response.headers = {"x-goog-upload-url": "https://upload.example.com/upload123"}
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
