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
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
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


@pytest.fixture
def mock_request():
    request = Mock()
    request.method = "POST"
    request.headers = {}
    request.url = httpx.URL("http://test.com/test")
    return request


@pytest.fixture
def mock_response():
    return Mock()


@pytest.fixture
def mock_user_api_key_dict():
    return {"api_key": "test-key"}


@patch("litellm.utils.get_secret")
@patch(
    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.create_pass_through_route"
)
@pytest.mark.asyncio()
async def test_assemblyai_proxy_route_basic_post(
    mock_create_route,
    mock_get_secret,
    mock_request,
    mock_response,
    mock_user_api_key_dict,
):
    """Test basic POST request handling for AssemblyAI proxy route"""
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        assemblyai_proxy_route,
    )

    # Setup mocks
    mock_get_secret.return_value = "test-assemblyai-key"
    mock_request.json = AsyncMock(return_value={"text": "test"})
    mock_endpoint_func = AsyncMock(return_value={"result": "success"})
    mock_create_route.return_value = mock_endpoint_func

    result = await assemblyai_proxy_route(
        endpoint="v2/transcript",
        request=mock_request,
        fastapi_response=mock_response,
        user_api_key_dict=mock_user_api_key_dict,
    )

    assert result == {"result": "success"}
    mock_create_route.assert_called_once_with(
        endpoint="v2/transcript",
        target="https://api.assemblyai.com/v2/transcript",
        custom_headers={"Authorization": "test-assemblyai-key"},
    )


@patch("litellm.utils.get_secret")
@patch(
    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.create_pass_through_route"
)
@pytest.mark.asyncio()
async def test_assemblyai_proxy_route_get_transcript(
    mock_create_route,
    mock_get_secret,
    mock_request,
    mock_response,
    mock_user_api_key_dict,
):
    """Test GET request handling for retrieving a specific transcript from AssemblyAI"""
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        assemblyai_proxy_route,
    )

    # Setup mocks
    mock_get_secret.return_value = "test-assemblyai-key"
    mock_request.method = "GET"
    mock_endpoint_func = AsyncMock(
        return_value={"id": "test-transcript-id", "status": "completed"}
    )
    mock_create_route.return_value = mock_endpoint_func

    result = await assemblyai_proxy_route(
        endpoint="v2/transcript/test-transcript-id",
        request=mock_request,
        fastapi_response=mock_response,
        user_api_key_dict=mock_user_api_key_dict,
    )

    assert result == {"id": "test-transcript-id", "status": "completed"}
    mock_create_route.assert_called_once_with(
        endpoint="v2/transcript/test-transcript-id",
        target="https://api.assemblyai.com/v2/transcript/test-transcript-id",
        custom_headers={"Authorization": "test-assemblyai-key"},
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
