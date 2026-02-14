"""
Test for Bedrock ChecksumMismatch error handling.

When Bedrock returns a JSON error response instead of a binary event stream,
the botocore EventStreamBuffer throws ChecksumMismatch. This test verifies
that we properly catch this error and return a meaningful error message.

Related issue: https://github.com/BerriAI/litellm/issues/20589
"""

import pytest

from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
from litellm.llms.bedrock.common_utils import BedrockError


class TestBedrockChecksumMismatchHandling:
    """Test that ChecksumMismatch errors from Bedrock are handled properly."""

    def test_iter_bytes_handles_json_error_response(self):
        """
        Test that iter_bytes properly handles when Bedrock returns a JSON error
        instead of a binary event stream.
        """
        decoder = AWSEventStreamDecoder(model="anthropic.claude-3-sonnet-20240229-v1:0")

        # Simulate Bedrock returning a JSON error response
        # This is what happens with inference profile ARN format issues
        json_error = b'{"message": "Validation error: Invalid inference profile ARN format"}'

        def mock_iterator():
            yield json_error

        # Should raise BedrockError with the actual error message, not ChecksumMismatch
        with pytest.raises(BedrockError) as exc_info:
            list(decoder.iter_bytes(mock_iterator()))

        assert "Bedrock returned error" in str(exc_info.value.message)
        assert "Validation error" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_aiter_bytes_handles_json_error_response(self):
        """
        Test that aiter_bytes properly handles when Bedrock returns a JSON error
        instead of a binary event stream (async version).
        """
        decoder = AWSEventStreamDecoder(model="anthropic.claude-3-sonnet-20240229-v1:0")

        # Simulate Bedrock returning a JSON error response
        json_error = b'{"message": "Validation error: Invalid inference profile ARN format"}'

        async def mock_async_iterator():
            yield json_error

        # Should raise BedrockError with the actual error message
        with pytest.raises(BedrockError) as exc_info:
            chunks = []
            async for chunk in decoder.aiter_bytes(mock_async_iterator()):
                chunks.append(chunk)

        assert "Bedrock returned error" in str(exc_info.value.message)
        assert "Validation error" in str(exc_info.value.message)

    def test_iter_bytes_handles_malformed_response(self):
        """
        Test that iter_bytes properly handles completely malformed responses
        that are neither valid event stream nor valid JSON.
        """
        decoder = AWSEventStreamDecoder(model="anthropic.claude-3-sonnet-20240229-v1:0")

        # Random bytes that aren't valid event stream or JSON
        malformed_data = b'\x00\x01\x02\x03invalid data'

        def mock_iterator():
            yield malformed_data

        with pytest.raises(BedrockError) as exc_info:
            list(decoder.iter_bytes(mock_iterator()))

        assert "malformed response" in str(exc_info.value.message).lower()
