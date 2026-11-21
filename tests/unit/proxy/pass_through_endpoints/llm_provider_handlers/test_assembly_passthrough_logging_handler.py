import asyncio
from unittest.mock import patch

import pytest

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
    AssemblyAITranscriptResponse,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


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


def test_get_assembly_transcript(assembly_handler, mock_transcript_response):
    """
    Test that the _get_assembly_transcript method calls GET /v2/transcript/{transcript_id}
    and uses the test key returned by the mocked get_credentials.
    """
    # Patch get_credentials to return "test-key"
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
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


def test_poll_assembly_for_transcript_response(assembly_handler, mock_transcript_response):
    """
    Test that the _poll_assembly_for_transcript_response method returns the correct transcript response
    """
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with patch("httpx.get") as mock_get:
            mock_get.return_value.json.return_value = mock_transcript_response
            mock_get.return_value.raise_for_status.return_value = None

            # Override polling settings for faster test
            assembly_handler.polling_interval = 0.01
            assembly_handler.max_polling_attempts = 2

            transcript = assembly_handler._poll_assembly_for_transcript_response(
                "test-transcript-id",
            )
            assert transcript == AssemblyAITranscriptResponse(**mock_transcript_response)


def test_is_assemblyai_route():
    """
    Test that the is_assemblyai_route method correctly identifies AssemblyAI routes
    """
    handler = PassThroughEndpointLogging()

    # Test positive cases
    assert handler.is_assemblyai_route("https://api.assemblyai.com/v2/transcript") == True
    assert handler.is_assemblyai_route("https://api.assemblyai.com/other/path") == True
    assert handler.is_assemblyai_route("https://api.assemblyai.com/transcript") == True

    # Test negative cases
    assert handler.is_assemblyai_route("https://example.com/other") == False
    assert handler.is_assemblyai_route("https://api.openai.com/v1/chat/completions") == False
    assert handler.is_assemblyai_route("") == False


# --- Security: SSRF via transcript_id path traversal ---


def test_get_assembly_transcript_rejects_slash_in_id(assembly_handler):
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("../../admin/credentials")


def test_get_assembly_transcript_rejects_dotdot_in_id(assembly_handler):
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("..evil")


def test_get_assembly_transcript_rejects_fragment_in_id(assembly_handler):
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("abc#suffix")


def test_get_assembly_transcript_rejects_query_in_id(assembly_handler):
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with pytest.raises(ValueError, match="disallowed characters"):
            assembly_handler._get_assembly_transcript("abc?x=1")


def test_get_assembly_transcript_allows_valid_id(assembly_handler, mock_transcript_response):
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with patch("httpx.get") as mock_get:
            mock_get.return_value.json.return_value = mock_transcript_response
            mock_get.return_value.raise_for_status.return_value = None

            transcript = assembly_handler._get_assembly_transcript("abc123-valid-id_xyz")
            assert transcript == mock_transcript_response
            called_url = mock_get.call_args[0][0]
            assert "abc123-valid-id_xyz" in called_url
            assert ".." not in called_url


def test_handle_assemblyai_passthrough_logging_resolves_model_from_transcript_response(assembly_handler):
    """
    Test that when the initial response body does not contain speech_model,
    it resolves model from transcript_response and records it in kwargs and logging_obj.
    """
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    initial_response = {"id": "test-transcript-id", "status": "queued"}
    polled_response = {
        "id": "test-transcript-id",
        "speech_model": "best",
        "status": "completed",
        "audio_duration": 120.0,
    }

    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_trace_id = "test-trace-id"
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.cost_breakdown = None

    with patch.object(
        assembly_handler,
        "_poll_assembly_for_transcript_response",
        return_value=AssemblyAITranscriptResponse(**polled_response),
    ):
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler.get_standard_logging_object_payload",
            return_value={},
        ):
            with patch("asyncio.run"):
                assembly_handler._handle_assemblyai_passthrough_logging(
                    httpx_response=MagicMock(),
                    response_body=initial_response,
                    logging_obj=mock_logging_obj,
                    url_route="/assemblyai/v2/transcript",
                    result="success",
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    cache_hit=False,
                )

                assert mock_logging_obj.model_call_details["model"] == "best"
                assert mock_logging_obj.model_call_details["response_cost"] is not None
                assert mock_logging_obj.model_call_details["response_cost"] > 0


def test_handle_assemblyai_passthrough_logging_preserves_explicit_speech_model(assembly_handler):
    """
    Test that when the initial response body already has speech_model,
    it is preserved and passed to cost calculation and logging.
    """
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    initial_response = {"id": "test-nano-id", "speech_model": "nano", "status": "queued"}
    polled_response = {
        "id": "test-nano-id",
        "speech_model": "nano",
        "status": "completed",
        "audio_duration": 60.0,
    }

    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {}

    with patch.object(
        assembly_handler,
        "_poll_assembly_for_transcript_response",
        return_value=AssemblyAITranscriptResponse(**polled_response),
    ):
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler.get_standard_logging_object_payload",
            return_value={},
        ):
            with patch("asyncio.run"):
                assembly_handler._handle_assemblyai_passthrough_logging(
                    httpx_response=MagicMock(),
                    response_body=initial_response,
                    logging_obj=mock_logging_obj,
                    url_route="/assemblyai/v2/transcript",
                    result="success",
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    cache_hit=False,
                )

                assert mock_logging_obj.model_call_details["model"] == "nano"
                assert mock_logging_obj.model_call_details["response_cost"] is not None


def test_get_cost_per_second_for_assembly_model_fallback():
    """
    Test that get_cost_per_second_for_assembly_model falls back to best when unknown model is used.
    """
    best_cost = AssemblyAIPassthroughLoggingHandler.get_cost_per_second_for_assembly_model("best")
    fallback_cost = AssemblyAIPassthroughLoggingHandler.get_cost_per_second_for_assembly_model("unknown-model")
    assert best_cost is not None
    assert fallback_cost == best_cost


@pytest.fixture(autouse=True)
async def _drain_logging_worker():
    """
    The logging queue is bound to the running loop, so anything left queued when a test's loop
    goes away is carried onto the next loop and fires against that test's callbacks.
    """
    GLOBAL_LOGGING_WORKER.start()
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    except asyncio.TimeoutError:
        pass
    await GLOBAL_LOGGING_WORKER.stop()
    yield


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)
