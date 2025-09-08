import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))
from litellm.llms.sagemaker.common_utils import AWSEventStreamDecoder
from litellm.llms.sagemaker.completion.transformation import SagemakerConfig


@pytest.mark.asyncio
async def test_aiter_bytes_unicode_decode_error():
    """
    Test that AWSEventStreamDecoder.aiter_bytes() does not raise an error when encountering invalid UTF-8 bytes. (UnicodeDecodeError)


    Ensures stream processing continues despite the error.

    Relevant issue: https://github.com/BerriAI/litellm/issues/9165
    """
    # Create an instance of AWSEventStreamDecoder
    decoder = AWSEventStreamDecoder(model="test-model")

    # Create a mock event that will trigger a UnicodeDecodeError
    mock_event = MagicMock()
    mock_event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": b"\xff\xfe",  # Invalid UTF-8 bytes
    }

    # Create a mock EventStreamBuffer that yields our mock event
    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [mock_event]

    # Mock the EventStreamBuffer class
    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):
        # Create an async generator that yields some test bytes
        async def mock_iterator():
            yield b""

        # Process the stream
        chunks = []
        async for chunk in decoder.aiter_bytes(mock_iterator()):
            if chunk is not None:
                print("chunk=", chunk)
                chunks.append(chunk)

        # Verify that processing continued despite the error
        # The chunks list should be empty since we only sent invalid data
        assert len(chunks) == 0


@pytest.mark.asyncio
async def test_aiter_bytes_valid_chunk_followed_by_unicode_error():
    """
    Test that valid chunks are processed correctly even when followed by Unicode decode errors.
    This ensures errors don't corrupt or prevent processing of valid data that came before.

    Relevant issue: https://github.com/BerriAI/litellm/issues/9165
    """
    decoder = AWSEventStreamDecoder(model="test-model")

    # Create two mock events - first valid, then invalid
    mock_valid_event = MagicMock()
    mock_valid_event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": json.dumps({"token": {"text": "hello"}}).encode(),  # Valid data first
    }

    mock_invalid_event = MagicMock()
    mock_invalid_event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": b"\xff\xfe",  # Invalid UTF-8 bytes second
    }

    # Create a mock EventStreamBuffer that yields valid event first, then invalid
    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [mock_valid_event, mock_invalid_event]

    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):

        async def mock_iterator():
            yield b"test_bytes"

        chunks = []
        async for chunk in decoder.aiter_bytes(mock_iterator()):
            if chunk is not None:
                chunks.append(chunk)

        # Verify we got our valid chunk despite the subsequent error
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello"  # Verify the content of the valid chunk


class TestSagemakerTransform:
    def setup_method(self):
        self.config = SagemakerConfig()
        self.model = "test"
        self.logging_obj = MagicMock()

    def test_map_mistral_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_new_tokens": 256}

    def test_mistral_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_new_tokens": 200}
