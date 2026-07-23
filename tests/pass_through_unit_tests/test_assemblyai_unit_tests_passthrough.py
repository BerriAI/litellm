import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
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


def _mock_llm_passthrough_endpoint_router(api_key: str = "test-key"):
    mock_module = ModuleType(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
    )
    mock_module.passthrough_endpoint_router = SimpleNamespace(
        get_credentials=lambda **_: api_key
    )
    return patch.dict(
        sys.modules,
        {
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints": mock_module
        },
    )


def test_should_log_request():
    handler = AssemblyAIPassthroughLoggingHandler()
    assert handler._should_log_request("POST")
    assert not handler._should_log_request("GET")


def test_get_assembly_transcript(assembly_handler, mock_transcript_response):
    """
    Test that the _get_assembly_transcript method calls GET /v2/transcript/{transcript_id}
    and uses the test key returned by the mocked get_credentials.
    """
    # Patch get_credentials to return "test-key"
    with _mock_llm_passthrough_endpoint_router():
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
    with _mock_llm_passthrough_endpoint_router():
        with patch("httpx.get") as mock_get:
            mock_get.return_value.json.return_value = mock_transcript_response
            mock_get.return_value.raise_for_status.return_value = None

            # Override polling settings for faster test
            assembly_handler.polling_interval = 0.01
            assembly_handler.max_polling_attempts = 2

            transcript = assembly_handler._poll_assembly_for_transcript_response(
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
    assert handler.is_assemblyai_route("https://api.assemblyai.com/v2/transcript")
    assert handler.is_assemblyai_route("https://api.assemblyai.com/other/path")
    assert handler.is_assemblyai_route("https://api.assemblyai.com/transcript")

    # Test negative cases
    assert not handler.is_assemblyai_route("https://example.com/other")
    assert not handler.is_assemblyai_route(
        "https://api.openai.com/v1/chat/completions"
    )
    assert not handler.is_assemblyai_route("")


def test_get_assembly_region_from_url_returns_eu_for_proxy_path():
    handler = AssemblyAIPassthroughLoggingHandler()

    assert (
        handler._get_assembly_region_from_url(
            "https://proxy.company.com/eu.assemblyai/v2/transcript"
        )
        == "eu"
    )


def test_get_assembly_region_from_url_returns_eu_for_root_path_proxy_path():
    handler = AssemblyAIPassthroughLoggingHandler()

    assert (
        handler._get_assembly_region_from_url(
            "https://proxy.company.com/litellm/eu.assemblyai/v2/transcript"
        )
        == "eu"
    )


def test_get_assembly_region_from_url_returns_eu_for_api_eu_host():
    handler = AssemblyAIPassthroughLoggingHandler()

    assert (
        handler._get_assembly_region_from_url(
            "https://api.eu.assemblyai.com/v2/transcript"
        )
        == "eu"
    )


def test_get_assembly_region_from_url_returns_none_for_default_route():
    handler = AssemblyAIPassthroughLoggingHandler()

    assert (
        handler._get_assembly_region_from_url(
            "https://proxy.company.com/assemblyai/v2/transcript"
        )
        is None
    )


# --- Security: SSRF via transcript_id path traversal ---


def test_get_assembly_transcript_rejects_slash_in_id(assembly_handler):
    with _mock_llm_passthrough_endpoint_router():
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("../../admin/credentials")


def test_get_assembly_transcript_rejects_dotdot_in_id(assembly_handler):
    with _mock_llm_passthrough_endpoint_router():
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("..evil")


def test_get_assembly_transcript_rejects_fragment_in_id(assembly_handler):
    with _mock_llm_passthrough_endpoint_router():
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("abc#suffix")


def test_get_assembly_transcript_rejects_query_in_id(assembly_handler):
    with _mock_llm_passthrough_endpoint_router():
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("abc?x=1")


def test_get_assembly_transcript_allows_valid_id(
    assembly_handler, mock_transcript_response
):
    with _mock_llm_passthrough_endpoint_router():
        with patch("httpx.get") as mock_get:
            mock_get.return_value.json.return_value = mock_transcript_response
            mock_get.return_value.raise_for_status.return_value = None

            transcript = assembly_handler._get_assembly_transcript(
                "abc123-valid-id_xyz"
            )
            assert transcript == mock_transcript_response
            called_url = mock_get.call_args[0][0]
            assert "abc123-valid-id_xyz" in called_url
            assert ".." not in called_url
