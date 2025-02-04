import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
    AssemblyAITranscriptResponse,
)


@pytest.fixture
def assembly_handler():
    handler = AssemblyAIPassthroughLoggingHandler()
    handler.assemblyai_api_key = "test-key"
    return handler


@pytest.fixture
def mock_transcript_response():
    return {
        "id": "test-transcript-id",
        "language_model": "default",
        "acoustic_model": "default",
        "language_code": "en",
        "status": "completed",
        "audio_duration": 100.0,
    }


def test_should_log_request():
    handler = AssemblyAIPassthroughLoggingHandler()
    assert handler._should_log_request("POST") == True
    assert handler._should_log_request("GET") == False


def test_get_assembly_transcript(assembly_handler, mock_transcript_response):
    """
    Test that the _get_assembly_transcript method calls GET /v2/transcript/{transcript_id}
    """
    with patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = mock_transcript_response
        mock_get.return_value.raise_for_status.return_value = None

        transcript = assembly_handler._get_assembly_transcript("test-transcript-id")
        assert transcript == mock_transcript_response

        mock_get.assert_called_once_with(
            "https://api.assemblyai.com/v2/transcript/test-transcript-id",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )


def test_poll_assembly_for_transcript_response(
    assembly_handler, mock_transcript_response
):
    """
    Test that the _poll_assembly_for_transcript_response method returns the correct transcript response
    """
    with patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = mock_transcript_response
        mock_get.return_value.raise_for_status.return_value = None

        # Override polling settings for faster test
        assembly_handler.polling_interval = 0.01
        assembly_handler.max_polling_attempts = 2

        transcript = assembly_handler._poll_assembly_for_transcript_response(
            "test-transcript-id"
        )
        assert transcript == AssemblyAITranscriptResponse(**mock_transcript_response)
