from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.azure_speech_passthrough_logging_handler import (
    AzureSpeechPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)

SHORT_AUDIO_URL = "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US"
BATCH_URL = "https://eastus.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
TRANSCRIPT = '{"RecognitionStatus":"Success","DisplayText":"Hello world."}'


def _make_response(url: str) -> httpx.Response:
    request = httpx.Request("POST", url, headers={"Ocp-Apim-Subscription-Key": "server-secret"})
    return httpx.Response(200, request=request, text=TRANSCRIPT)


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.model_call_details = {}
    return logging_obj


class TestAzureSpeechPassthroughHandler:
    @pytest.mark.parametrize(
        "url_route,expected_model",
        [
            (SHORT_AUDIO_URL, "azure_speech/short-audio"),
            (BATCH_URL, "azure_speech/batch-transcription"),
            (f"{BATCH_URL}/8a5d3f2c-0b1e-4c7d-9e6f-1234567890ab/files", "azure_speech/batch-transcription"),
        ],
    )
    def test_records_model_provider_and_zero_cost(self, url_route: str, expected_model: str):
        logging_obj = _make_logging_obj()

        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(url_route),
            logging_obj=logging_obj,
            url_route=url_route,
            result=TRANSCRIPT,
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={},
        )

        assert handler_result["result"] == {"response": TRANSCRIPT}
        assert handler_result["kwargs"]["model"] == expected_model
        assert handler_result["kwargs"]["custom_llm_provider"] == "azure_speech"
        assert handler_result["kwargs"]["response_cost"] == 0.0
        assert handler_result["kwargs"]["standard_logging_object"]["response_cost"] == 0.0
        assert handler_result["kwargs"]["standard_logging_object"]["model"] == expected_model
        assert logging_obj.model_call_details["model"] == expected_model
        assert logging_obj.model_call_details["custom_llm_provider"] == "azure_speech"
        assert logging_obj.model_call_details["response_cost"] == 0.0

    def test_subscription_key_never_reaches_the_logging_payload(self):
        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(SHORT_AUDIO_URL),
            logging_obj=_make_logging_obj(),
            url_route=SHORT_AUDIO_URL,
            result=TRANSCRIPT,
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={},
        )

        assert "server-secret" not in repr(handler_result)


class TestIsAzureSpeechRoute:
    def test_matches_by_provider_tag(self):
        assert PassThroughEndpointLogging().is_azure_speech_route("azure_speech")

    @pytest.mark.parametrize("provider", ["azure", "azure_ai", "comprehendmedical", None])
    def test_does_not_match_other_providers(self, provider: str | None):
        assert not PassThroughEndpointLogging().is_azure_speech_route(provider)

    def test_config_driven_passthrough_to_azure_speech_host_is_not_claimed(self):
        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response(SHORT_AUDIO_URL),
            response_body={"RecognitionStatus": "Success"},
            request_body={},
            logging_obj=_make_logging_obj(),
            url_route=SHORT_AUDIO_URL,
            result=TRANSCRIPT,
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider=None,
        )

        assert normalized["kwargs"].get("model") != "azure_speech/short-audio"
        assert "response_cost" not in normalized["kwargs"]


class TestNormalizeDispatch:
    def test_normalize_routes_to_azure_speech_handler(self):
        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response(SHORT_AUDIO_URL),
            response_body={"RecognitionStatus": "Success"},
            request_body={},
            logging_obj=_make_logging_obj(),
            url_route=SHORT_AUDIO_URL,
            result=TRANSCRIPT,
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="azure_speech",
        )

        assert normalized["standard_logging_response_object"] == {"response": TRANSCRIPT}
        assert normalized["kwargs"]["model"] == "azure_speech/short-audio"
        assert normalized["kwargs"]["custom_llm_provider"] == "azure_speech"
        assert normalized["kwargs"]["response_cost"] == 0.0
