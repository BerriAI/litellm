import io
import json
import os
import sys
from unittest.mock import MagicMock, patch
from typing import Any

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
                                {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
                                {"word": "this", "start": 0.6, "end": 0.8, "confidence": 0.98},
                                {"word": "is", "start": 0.9, "end": 1.1, "confidence": 0.97},
                                {"word": "a", "start": 1.2, "end": 1.3, "confidence": 0.96},
                                {"word": "test", "start": 1.4, "end": 1.8, "confidence": 0.95},
                                {"word": "transcription", "start": 1.9, "end": 2.8, "confidence": 0.94},
                            ]
                        }
                    ]
                }
            ]
        }
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

    def test_basic_transcription(self, mock_deepgram_response, test_audio_bytes):
        """Test basic transcription without additional parameters"""
        
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
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL (should be basic URL without extra params)
            expected_url = "https://api.deepgram.com/v1/listen?model=nova-2"
            assert call_kwargs["url"] == expected_url
            
            # Verify headers
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Token test-api-key"
            
            # Verify request data is the audio bytes
            assert call_kwargs["data"] == test_audio_bytes
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."
            assert hasattr(response, '_hidden_params')

    def test_transcription_with_punctuate(self, mock_deepgram_response, test_audio_bytes):
        """Test transcription with punctuate=true parameter"""
        
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
                punctuate=True,
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains punctuate parameter
            expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true"
            assert call_kwargs["url"] == expected_url
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_diarize(self, mock_deepgram_response, test_audio_bytes):
        """Test transcription with diarize=true parameter"""
        
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
                diarize=True,
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains diarize parameter
            expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&diarize=true"
            assert call_kwargs["url"] == expected_url
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_measurements(self, mock_deepgram_response, test_audio_bytes):
        """Test transcription with measurements=true parameter"""
        
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
                measurements=True,
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains measurements parameter
            expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&measurements=true"
            assert call_kwargs["url"] == expected_url
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_multiple_params(self, mock_deepgram_response, test_audio_bytes):
        """Test transcription with multiple query parameters"""
        
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
                punctuate=True,
                diarize=False,
                measurements=True,
                smart_format=True,
                tier="enhanced",
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains all parameters
            url = call_kwargs["url"]
            assert "model=nova-2" in url
            assert "punctuate=true" in url
            assert "diarize=false" in url
            assert "measurements=true" in url
            assert "smart_format=true" in url
            assert "tier=enhanced" in url
            assert url.startswith("https://api.deepgram.com/v1/listen?")
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_language_param(self, mock_deepgram_response, test_audio_bytes):
        """Test that language parameter is handled separately (not in URL)"""
        
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
                language="en",
                punctuate=True,
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains punctuate but NOT language
            expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true"
            assert call_kwargs["url"] == expected_url
            assert "language=" not in call_kwargs["url"]
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_custom_api_base(self, mock_deepgram_response, test_audio_bytes):
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
            expected_url = "https://custom.deepgram.com/v2/listen?model=nova-2&punctuate=true"
            assert call_kwargs["url"] == expected_url
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    @pytest.mark.asyncio
    async def test_async_transcription_with_params(self, mock_deepgram_response, test_audio_bytes):
        """Test async transcription with query parameters"""
        
        mock_response = MagicMock()
        mock_response.json.return_value = mock_deepgram_response
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=mock_response,
        ) as mock_post:
            
            response: TranscriptionResponse = await litellm.atranscription(
                model="deepgram/nova-2",
                file=test_audio_bytes,
                api_key="test-api-key",
                punctuate=True,
                diarize=True,
                measurements=False,
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains all parameters
            url = call_kwargs["url"]
            assert "model=nova-2" in url
            assert "punctuate=true" in url
            assert "diarize=true" in url
            assert "measurements=false" in url
            assert url.startswith("https://api.deepgram.com/v1/listen?")
            
            # Verify headers
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Token test-api-key"
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_with_file_object(self, mock_deepgram_response, test_audio_file):
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
            expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true"
            assert call_kwargs["url"] == expected_url
            
            # Verify request data contains the audio bytes
            assert call_kwargs["data"] == b"fake_audio_data_for_testing"
            
            # Verify response
            assert response.text == "Hello, this is a test transcription."

    def test_transcription_boolean_conversion(self, mock_deepgram_response, test_audio_bytes):
        """Test that boolean values are correctly converted to lowercase strings"""
        
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
                punctuate=True,  # Should become "true"
                diarize=False,   # Should become "false"
            )
            
            # Verify the HTTP call was made
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            
            # Verify URL contains correctly formatted boolean values
            url = call_kwargs["url"]
            assert "punctuate=true" in url  # lowercase "true"
            assert "diarize=false" in url   # lowercase "false"
            # Should not contain Python-style "True" or "False"
            assert "True" not in url
            assert "False" not in url