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
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


@pytest.fixture
def assembly_handler():
    handler = AssemblyAIPassthroughLoggingHandler()
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


@pytest.mark.asyncio
async def test_get_assembly_transcript(assembly_handler, mock_transcript_response):
    """
    Test that the _get_assembly_transcript method calls GET /v2/transcript/{transcript_id}
    and uses the test key returned by the mocked get_credentials.
    """
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        mock_response = Mock()
        mock_response.json.return_value = mock_transcript_response
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        transcript = await assembly_handler._get_assembly_transcript(
            "test-transcript-id", client=mock_client
        )
        assert transcript == mock_transcript_response

        mock_client.get.assert_called_once_with(
            "https://api.assemblyai.com/v2/transcript/test-transcript-id",
            headers={
                "Authorization": "test-key",
                "Content-Type": "application/json",
            },
        )


@pytest.mark.asyncio
async def test_poll_assembly_for_transcript_response(
    assembly_handler, mock_transcript_response
):
    """
    Test that the _poll_assembly_for_transcript_response method returns the correct transcript response
    """
    with patch.object(
        assembly_handler, "_get_assembly_transcript",
        new_callable=AsyncMock,
        return_value=mock_transcript_response,
    ):
        assembly_handler.polling_interval = 0.01
        assembly_handler.max_polling_attempts = 2

        transcript = await assembly_handler._poll_assembly_for_transcript_response(
            "test-transcript-id",
        )
        assert transcript == AssemblyAITranscriptResponse(
            **mock_transcript_response
        )


def test_is_assemblyai_route():
    """
    Test that the is_assemblyai_route method correctly identifies AssemblyAI routes
    """
    handler = PassThroughEndpointLogging()

    # Test positive cases
    assert (
        handler.is_assemblyai_route("https://api.assemblyai.com/v2/transcript") == True
    )
    assert handler.is_assemblyai_route("https://api.assemblyai.com/other/path") == True
    assert handler.is_assemblyai_route("https://api.assemblyai.com/transcript") == True

    # Test negative cases
    assert handler.is_assemblyai_route("https://example.com/other") == False
    assert (
        handler.is_assemblyai_route("https://api.openai.com/v1/chat/completions")
        == False
    )
    assert handler.is_assemblyai_route("") == False


@pytest.mark.asyncio
async def test_assemblyai_handler_runs_in_async_context():
    """
    The handler must be awaitable and run in the caller's async context
    (not a thread pool) so _handle_logging() can access the database.
    """
    handler = AssemblyAIPassthroughLoggingHandler()

    mock_httpx_response = Mock(spec=httpx.Response)
    mock_httpx_response.request = Mock()
    mock_httpx_response.request.method = "POST"
    mock_httpx_response.text = '{"id": "test-id", "status": "queued"}'

    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    response_body = {"id": "test-id", "status": "queued", "speech_model": "nano"}
    mock_transcript = {"id": "test-id", "status": "completed", "audio_duration": 60.0}

    with patch.object(
        handler, "_poll_assembly_for_transcript_response",
        new_callable=AsyncMock,
        return_value=AssemblyAITranscriptResponse(**mock_transcript),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler.get_standard_logging_object_payload",
        return_value={},
    ), patch(
        "litellm.proxy.pass_through_endpoints.success_handler.PassThroughEndpointLogging._handle_logging",
        new_callable=AsyncMock,
    ) as mock_handle_logging:
        # This must be awaitable (async), not fire-and-forget (thread pool)
        await handler.assemblyai_passthrough_logging_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route="https://api.assemblyai.com/v2/transcript",
            result="{}",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
        )

        # response_cost should be set from dynamic calculation
        assert mock_logging_obj.model_call_details.get("response_cost") is not None
