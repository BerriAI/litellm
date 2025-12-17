"""
Tests for IBM WatsonX Audio Transcription.

Validates that litellm.transcription transforms requests correctly for WatsonX.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm


class TestWatsonXAudioTranscription:
    """Tests for WatsonX audio transcription via litellm.transcription."""

    @pytest.mark.asyncio
    async def test_watsonx_transcription_url_and_headers(self):
        """
        Test that litellm.transcription sends request to correct WatsonX URL with proper headers.
        """
        captured_request = {}

        async def mock_post(*args, **kwargs):
            captured_request["url"] = str(kwargs.get("url", args[0] if args else None))
            captured_request["headers"] = kwargs.get("headers", {})
            captured_request["data"] = kwargs.get("data", {})
            captured_request["files"] = kwargs.get("files", {})
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "text": "test transcription",
                "duration": 1.0,
            }
            mock_response.status_code = 200
            return mock_response

        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new=mock_post):
            try:
                await litellm.atranscription(
                    model="watsonx/whisper-large-v3-turbo",
                    file=b"fake_audio_data",
                    api_base="https://us-south.ml.cloud.ibm.com",
                    api_key="test-api-key",
                    project_id="test-project-123",
                    token="test-bearer-token",
                )
            except Exception:
                pass  # We just want to capture the request

        # Validate URL contains WatsonX audio transcription endpoint
        assert "/ml/v1/audio/transcriptions" in captured_request["url"]
        assert "version=" in captured_request["url"]
        # project_id should NOT be in URL (it should be in form data instead)
        assert "project_id=test-project-123" not in captured_request["url"]

        # Validate headers contain WatsonX auth
        assert "Authorization" in captured_request["headers"]
        assert "Bearer test-bearer-token" in captured_request["headers"]["Authorization"]
        
        # Validate Content-Type is NOT set (httpx sets multipart/form-data automatically)
        assert "Content-Type" not in captured_request["headers"]
        
        # Validate project_id is in form data, not URL
        assert captured_request["data"].get("project_id") == "test-project-123"
        
        # Validate file is in files dict
        assert "file" in captured_request["files"]

    @pytest.mark.asyncio
    async def test_watsonx_transcription_request_body(self):
        """
        Test that litellm.transcription sends correct request body for WatsonX.
        
        Validates that:
        - Request uses multipart/form-data (data + files)
        - Model name has watsonx/ prefix removed
        - project_id is in form data, not URL
        - Audio file is in files dict
        - OpenAI params are included in form data
        """
        captured_request = {}

        async def mock_post(*args, **kwargs):
            captured_request["data"] = kwargs.get("data", {})
            captured_request["files"] = kwargs.get("files", {})
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "text": "test transcription",
                "duration": 1.0,
            }
            mock_response.status_code = 200
            return mock_response

        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new=mock_post):
            try:
                await litellm.atranscription(
                    model="watsonx/whisper-large-v3-turbo",
                    file=b"fake_audio_data",
                    api_base="https://us-south.ml.cloud.ibm.com",
                    api_key="test-api-key",
                    project_id="test-project-123",
                    token="test-bearer-token",
                    language="en",
                    temperature=0.5,
                )
            except Exception:
                pass  # We just want to capture the request

        # Validate form data contains expected fields
        data = captured_request.get("data", {})

        print("JSON DUMPS captured_request:")
        print(json.dumps(captured_request, indent=4, default=str))
        
        # Model name should NOT have watsonx/ prefix
        assert data.get("model") == "whisper-large-v3-turbo"
        
        # project_id should be in form data
        assert data.get("project_id") == "test-project-123"
        
        # OpenAI params should be in form data
        assert data.get("language") == "en"
        assert data.get("temperature") == 0.5
        # response_format should NOT be set by default - only send what user specifies
        assert "response_format" not in data
        
        # Validate file is in files dict (multipart/form-data)
        files = captured_request.get("files", {})
        assert "file" in files
        assert isinstance(files["file"], tuple)  # Should be (filename, content, content_type)

    @pytest.mark.asyncio
    async def test_watsonx_transcription_only_user_params_sent(self):
        """
        Test that only user-specified params are sent in request body to WatsonX.
        
        LiteLLM should NOT add extra params like response_format if user didn't specify them.
        """
        captured_request = {}

        async def mock_post(*args, **kwargs):
            captured_request["data"] = kwargs.get("data", {})
            captured_request["files"] = kwargs.get("files", {})
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "text": "test transcription",
                "duration": 1.0,
            }
            mock_response.status_code = 200
            return mock_response

        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new=mock_post):
            try:
                # Minimal request - only required params
                await litellm.atranscription(
                    model="watsonx/whisper-large-v3-turbo",
                    file=b"fake_audio_data",
                    api_base="https://us-south.ml.cloud.ibm.com",
                    api_key="test-api-key",
                    project_id="test-project-123",
                    token="test-bearer-token",
                )
            except Exception:
                pass  # We just want to capture the request

        data = captured_request.get("data", {})
        
        # These are the ONLY keys that should be in data
        expected_keys = {"model", "project_id"}
        actual_keys = set(data.keys())
        
        assert actual_keys == expected_keys, (
            f"Request body should only contain {expected_keys}, "
            f"but got {actual_keys}. "
            f"Extra keys: {actual_keys - expected_keys}"
        )
        
        # Specifically verify response_format is NOT added
        assert "response_format" not in data, "response_format should NOT be added by default"
        
        # Verify file is sent separately
        files = captured_request.get("files", {})
        assert "file" in files
