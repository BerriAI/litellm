"""
Test for issue #17209: Clearer error when LLM endpoint returns empty response
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai.common_utils import OpenAIError


class TestEmptyResponseHandling:
    """Test that empty/invalid responses from LLM endpoints produce clear error messages"""

    def test_sync_empty_string_response_raises_clear_error(self):
        """
        Test that when an OpenAI-compatible endpoint returns an empty string,
        we get a clear error instead of "'str' object has no attribute 'model_dump'"
        """
        openai_chat = OpenAIChatCompletion()

        # Mock the raw response to return an empty string from parse()
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = ""  # Empty string response

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)
        assert "Check the reverse proxy or model server configuration" in str(
            exc_info.value
        )

    def test_sync_none_response_raises_clear_error(self):
        """Test that None response also produces a clear error"""
        openai_chat = OpenAIChatCompletion()

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = None

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)

    def test_valid_response_passes_through(self):
        """Test that a valid response with model_dump passes through correctly"""
        openai_chat = OpenAIChatCompletion()

        # Create a mock response that has model_dump (like a real Pydantic model)
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"choices": []}

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}]},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert response == mock_response
        assert headers == {"x-request-id": "123"}

    def test_sync_streaming_response_passes_through_without_model_dump(self):
        """
        Test that streaming responses (which don't have model_dump) pass through
        correctly without raising an error. This validates the fix for VLLM streaming.
        """
        openai_chat = OpenAIChatCompletion()

        # Create a mock response WITHOUT model_dump (like an AsyncStream/Iterator)
        mock_stream = MagicMock(spec=[])  # spec=[] means no attributes

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = mock_stream

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        # Key: data has stream=True - this should bypass the model_dump check
        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}], "stream": True},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert response == mock_stream
        assert headers == {"x-request-id": "123"}


def _as_unicode_encode_error() -> UnicodeEncodeError:
    """A genuine UnicodeEncodeError, matching the one httpx raises encoding a non-ASCII header value."""
    try:
        "café".encode("ascii")
    except UnicodeEncodeError as e:
        return e
    raise AssertionError("expected 'café'.encode('ascii') to raise")


class TestNonAsciiHeaderHandling:
    """Test for issue #44830: a non-ASCII header value fails deep inside httpx with
    only an ascii-codec error and a byte offset; litellm should name the header."""

    def test_sync_non_ascii_extra_header_names_the_header(self):
        openai_chat = OpenAIChatCompletion()

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = _as_unicode_encode_error()

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={
                    "messages": [{"role": "user", "content": "test"}],
                    "extra_headers": {"x-trace-locale": "café"},
                },
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "extra_headers['x-trace-locale']" in str(exc_info.value)
        assert "café" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_non_ascii_extra_header_names_the_header(self):
        openai_chat = OpenAIChatCompletion()

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(
            side_effect=_as_unicode_encode_error()
        )

        with pytest.raises(OpenAIError) as exc_info:
            await openai_chat.make_openai_chat_completion_request(
                openai_aclient=mock_client,
                data={
                    "messages": [{"role": "user", "content": "test"}],
                    "extra_headers": {"x-trace-locale": "café"},
                },
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "extra_headers['x-trace-locale']" in str(exc_info.value)
        assert "café" not in str(exc_info.value)

    def test_sync_non_ascii_extra_header_does_not_leak_a_credential_header_value(self):
        """A configured header such as x-api-key may itself be the credential;
        naming it is fine, but the error must never echo its value back."""
        openai_chat = OpenAIChatCompletion()

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = _as_unicode_encode_error()

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={
                    "messages": [{"role": "user", "content": "test"}],
                    "extra_headers": {"x-api-key": "sk-live-secret-café"},
                },
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "extra_headers['x-api-key']" in str(exc_info.value)
        assert "sk-live-secret" not in str(exc_info.value)

    def test_sync_non_ascii_header_outside_extra_headers_names_no_header(self):
        """When none of this request's own extra_headers are at fault (e.g. the
        API key or organization id carried the non-ASCII character instead),
        the error says so rather than naming an innocent header."""
        openai_chat = OpenAIChatCompletion()

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = _as_unicode_encode_error()

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "None of this request's extra_headers are at fault" in str(exc_info.value)
        assert "api_key" in str(exc_info.value)
