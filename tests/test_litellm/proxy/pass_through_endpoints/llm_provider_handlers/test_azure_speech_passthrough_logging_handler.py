import json
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.azure_speech_passthrough_logging_handler import (
    AzureSpeechPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)

SHORT_AUDIO_URL = (
    "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US"
)
BATCH_URL = "https://eastus.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
FAST_URL = "https://eastus.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
FAST_BODY = {"durationMilliseconds": 5061, "combinedPhrases": [{"text": "Hello world."}]}
FAST_AUDIO_SECONDS = 5.061
TRANSCRIPT_BODY = {
    "RecognitionStatus": "Success",
    "Offset": 5000000,
    "Duration": 25000000,
    "DisplayText": "Hello world.",
}
TRANSCRIPT = json.dumps(TRANSCRIPT_BODY)
TRANSCRIPT_AUDIO_SECONDS = 3.0
PRICE_PER_SECOND = 0.5


@pytest.fixture(autouse=True)
def azure_stt_price(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/speech/azure-stt",
        {
            "litellm_provider": "azure",
            "mode": "audio_transcription",
            "input_cost_per_second": PRICE_PER_SECOND,
            "output_cost_per_second": 0.0,
        },
    )


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
        "url_route,expected_model,expected_cost",
        [
            (SHORT_AUDIO_URL, "azure_speech/short-audio", TRANSCRIPT_AUDIO_SECONDS * PRICE_PER_SECOND),
            (FAST_URL, "azure_speech/fast-transcription", FAST_AUDIO_SECONDS * PRICE_PER_SECOND),
            (BATCH_URL, "azure_speech/batch-transcription", 0.0),
            (f"{BATCH_URL}/8a5d3f2c-0b1e-4c7d-9e6f-1234567890ab/files", "azure_speech/batch-transcription", 0.0),
        ],
    )
    def test_records_model_provider_and_cost(self, url_route: str, expected_model: str, expected_cost: float):
        logging_obj = _make_logging_obj()

        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(url_route),
            response_body={**TRANSCRIPT_BODY, **FAST_BODY},
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
        assert handler_result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
        assert handler_result["kwargs"]["standard_logging_object"]["response_cost"] == pytest.approx(expected_cost)
        assert handler_result["kwargs"]["standard_logging_object"]["model"] == expected_model
        assert logging_obj.model_call_details["model"] == expected_model
        assert logging_obj.model_call_details["custom_llm_provider"] == "azure_speech"
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(expected_cost)

    @pytest.mark.parametrize(
        "response_body",
        [
            {"RecognitionStatus": "NoMatch", "Offset": 0, "Duration": 0},
            {"RecognitionStatus": "InitialSilenceTimeout"},
            {"Offset": "5000000", "Duration": "25000000"},
            {},
            [],
            None,
        ],
    )
    def test_short_audio_without_recognized_duration_logs_zero_cost(
        self, response_body: dict[str, object] | list[dict[str, object]] | None
    ):
        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(SHORT_AUDIO_URL),
            response_body=response_body,
            logging_obj=_make_logging_obj(),
            url_route=SHORT_AUDIO_URL,
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={},
        )

        assert handler_result["kwargs"]["model"] == "azure_speech/short-audio"
        assert handler_result["kwargs"]["response_cost"] == 0.0

    @pytest.mark.parametrize(
        "response_body",
        [{"durationMilliseconds": 0}, {"durationMilliseconds": "5061"}, {"duration": 5061}, {}, [], None],
    )
    def test_fast_transcription_without_duration_milliseconds_logs_zero_cost(
        self, response_body: dict[str, object] | list[dict[str, object]] | None
    ):
        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(FAST_URL),
            response_body=response_body,
            logging_obj=_make_logging_obj(),
            url_route=FAST_URL,
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={},
        )

        assert handler_result["kwargs"]["model"] == "azure_speech/fast-transcription"
        assert handler_result["kwargs"]["response_cost"] == 0.0

    def test_missing_price_entry_still_logs_the_row_at_zero_cost(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delitem(litellm.model_cost, "azure/speech/azure-stt")

        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(SHORT_AUDIO_URL),
            response_body=TRANSCRIPT_BODY,
            logging_obj=_make_logging_obj(),
            url_route=SHORT_AUDIO_URL,
            result=TRANSCRIPT,
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={},
        )

        assert handler_result["kwargs"]["model"] == "azure_speech/short-audio"
        assert handler_result["kwargs"]["custom_llm_provider"] == "azure_speech"
        assert handler_result["kwargs"]["response_cost"] == 0.0

    def test_subscription_key_never_reaches_the_logging_payload(self):
        handler_result = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
            httpx_response=_make_response(SHORT_AUDIO_URL),
            response_body=TRANSCRIPT_BODY,
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
            response_body=TRANSCRIPT_BODY,
            request_body={},
            logging_obj=_make_logging_obj(),
            url_route=SHORT_AUDIO_URL,
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="azure_speech",
        )

        assert normalized["standard_logging_response_object"] == {"response": ""}
        assert normalized["kwargs"]["model"] == "azure_speech/short-audio"
        assert normalized["kwargs"]["custom_llm_provider"] == "azure_speech"
        assert normalized["kwargs"]["response_cost"] == pytest.approx(TRANSCRIPT_AUDIO_SECONDS * PRICE_PER_SECOND)
