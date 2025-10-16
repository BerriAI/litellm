import io
import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.utils import TranscriptionResponse


@pytest.fixture
def mock_deepgram_response():
    """Mock Deepgram API response"""
    return {
        "metadata": {
            "transaction_key": "deprecated",
            "request_id": "test-request-id",
            "sha256": "test-sha",
            "created": "2024-01-01T00:00:00.000Z",
            "duration": 10.5,
            "channels": 1,
            "models": ["nova-2"],
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello, this is a test transcription.",
                            "confidence": 0.99,
                            "words": [
                                {
                                    "word": "Hello",
                                    "start": 0.0,
                                    "end": 0.5,
                                    "confidence": 0.99,
                                },
                                {
                                    "word": "this",
                                    "start": 0.6,
                                    "end": 0.8,
                                    "confidence": 0.98,
                                },
                                {
                                    "word": "is",
                                    "start": 0.9,
                                    "end": 1.1,
                                    "confidence": 0.97,
                                },
                                {
                                    "word": "a",
                                    "start": 1.2,
                                    "end": 1.3,
                                    "confidence": 0.96,
                                },
                                {
                                    "word": "test",
                                    "start": 1.4,
                                    "end": 1.8,
                                    "confidence": 0.95,
                                },
                                {
                                    "word": "transcription",
                                    "start": 1.9,
                                    "end": 2.8,
                                    "confidence": 0.94,
                                },
                            ],
                        }
                    ]
                }
            ]
        },
    }


@pytest.fixture
def test_audio_bytes():
    """Mock audio file bytes"""
    return b"fake_audio_data_for_testing"


@pytest.fixture
def test_audio_file():
    """Mock audio file object"""
    return io.BytesIO(b"fake_audio_data_for_testing")


class TestDeepgramMockTranscription:
    """Test Deepgram transcription with mocked HTTP requests"""

    @pytest.mark.parametrize(
        "optional_params,expected_url",
        [
            # Basic transcription without parameters
            ({}, "https://api.deepgram.com/v1/listen?model=nova-2"),
            # Single parameters
            (
                {"punctuate": True},
                "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true",
            ),
            (
                {"diarize": True},
                "https://api.deepgram.com/v1/listen?model=nova-2&diarize=true",
            ),
            (
                {"measurements": True},
                "https://api.deepgram.com/v1/listen?model=nova-2&measurements=true",
            ),
            (
                {"diarize": False},
                "https://api.deepgram.com/v1/listen?model=nova-2&diarize=false",
            ),
            # String parameters
            (
                {"tier": "enhanced"},
                "https://api.deepgram.com/v1/listen?model=nova-2&tier=enhanced",
            ),
            (
                {"version": "latest"},
                "https://api.deepgram.com/v1/listen?model=nova-2&version=latest",
            ),
            # Language parameter should be excluded
            (
                {"language": "en", "punctuate": True},
                "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true",
            ),
            # Multiple parameters with boolean conversion
            (
                {"punctuate": True, "diarize": False},
                "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&diarize=false",
            ),
            # Multiple mixed parameters
            (
                {
                    "punctuate": True,
                    "diarize": False,
                    "measurements": True,
                    "smart_format": True,
                    "tier": "enhanced",
                },
                None,
            ),  # We'll check contains for this one since order may vary
        ],
    )
    def test_transcription_url_generation(
        self,
        mock_deepgram_response,
        test_audio_bytes,
        optional_params,
        expected_url,
    ):
        """Test transcription URL generation with various parameters"""

        # Create mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_deepgram_response
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}

        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            response: TranscriptionResponse = litellm.transcription(
                model="deepgram/nova-2",
                file=test_audio_bytes,
                api_key="test-api-key",
                **optional_params,
            )

            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs

            # Verify URL
            actual_url = call_kwargs["url"]
            if expected_url is None:
                # For multiple params, check that all expected parts are present
                assert "model=nova-2" in actual_url
                assert "punctuate=true" in actual_url
                assert "diarize=false" in actual_url
                assert "measurements=true" in actual_url
                assert "smart_format=true" in actual_url
                assert "tier=enhanced" in actual_url
                assert actual_url.startswith("https://api.deepgram.com/v1/listen?")
                # Ensure language is not included even if it was in optional_params for other tests
                assert "language=" not in actual_url
            else:
                assert (
                    actual_url == expected_url
                ), f"Expected {expected_url}, got {actual_url}"

            # Verify headers
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Token test-api-key"

            # Verify response
            assert response.text == "Hello, this is a test transcription."
            assert hasattr(response, "_hidden_params")

    def test_transcription_with_custom_api_base(
        self, mock_deepgram_response, test_audio_bytes
    ):
        """Test transcription with custom API base URL"""

        mock_response = MagicMock()
        mock_response.json.return_value = mock_deepgram_response
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}

        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            response: TranscriptionResponse = litellm.transcription(
                model="deepgram/nova-2",
                file=test_audio_bytes,
                api_key="test-api-key",
                api_base="https://custom.deepgram.com/v2",
                punctuate=True,
            )

            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs

            # Verify custom API base is used
            expected_url = (
                "https://custom.deepgram.com/v2/listen?model=nova-2&punctuate=true"
            )
            assert call_kwargs["url"] == expected_url

            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_file_object(
        self, mock_deepgram_response, test_audio_file
    ):
        """Test transcription with file-like object"""

        mock_response = MagicMock()
        mock_response.json.return_value = mock_deepgram_response
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}

        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            response: TranscriptionResponse = litellm.transcription(
                model="deepgram/nova-2",
                file=test_audio_file,
                api_key="test-api-key",
                punctuate=True,
            )

            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs

            # Verify URL contains punctuate parameter
            expected_url = (
                "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true"
            )
            assert call_kwargs["url"] == expected_url

            # Verify response
            assert response.text == "Hello, this is a test transcription."
