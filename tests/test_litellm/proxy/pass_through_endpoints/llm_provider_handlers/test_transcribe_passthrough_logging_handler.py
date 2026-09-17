from datetime import datetime
from unittest.mock import MagicMock

import httpx

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.transcribe_passthrough_logging_handler import (
    TranscribePassthroughLoggingHandler,
    transcribe_supported_operations,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


def _make_response(operation: str) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "https://transcribe.us-west-2.amazonaws.com/",
        headers={"X-Amz-Target": f"Transcribe.{operation}"},
    )
    return httpx.Response(200, request=request, text='{"TranscriptionJob": {}}')


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.model_call_details = {}
    return logging_obj


class TestTranscribeSupportedOperations:
    def test_matches_the_installed_botocore_service_model(self):
        from botocore.session import get_session

        assert transcribe_supported_operations() == frozenset(
            get_session().get_service_model("transcribe").operation_names
        )


class TestTranscribePassthroughHandler:
    def test_records_model_provider_and_zero_cost(self):
        logging_obj = _make_logging_obj()
        request_body = {"TranscriptionJobName": "litellm-job-1"}

        handler_result = TranscribePassthroughLoggingHandler.transcribe_passthrough_handler(
            httpx_response=_make_response("StartTranscriptionJob"),
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body=request_body,
        )

        assert handler_result["result"] == {"response": '{"TranscriptionJob": {}}'}
        assert handler_result["kwargs"]["model"] == "transcribe/StartTranscriptionJob"
        assert handler_result["kwargs"]["custom_llm_provider"] == "transcribe"
        assert handler_result["kwargs"]["response_cost"] == 0.0
        assert "standard_logging_object" in handler_result["kwargs"]
        assert logging_obj.model_call_details["model"] == "transcribe/StartTranscriptionJob"
        assert logging_obj.model_call_details["custom_llm_provider"] == "transcribe"
        assert logging_obj.model_call_details["response_cost"] == 0.0
        assert request_body == {"TranscriptionJobName": "litellm-job-1"}


class TestIsTranscribeRoute:
    def test_matches_by_provider_tag(self):
        assert PassThroughEndpointLogging().is_transcribe_route("transcribe")

    def test_does_not_match_other_providers(self):
        assert not PassThroughEndpointLogging().is_transcribe_route("comprehendmedical")

    def test_dispatch_reaches_transcribe_handler(self):
        logging_obj = _make_logging_obj()

        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response("GetTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            request_body={"TranscriptionJobName": "litellm-job-1"},
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="transcribe",
        )

        assert normalized["kwargs"]["model"] == "transcribe/GetTranscriptionJob"
        assert normalized["kwargs"]["response_cost"] == 0.0

    def test_config_driven_passthrough_to_transcribe_host_is_not_claimed(self):
        logging_obj = _make_logging_obj()

        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response("GetTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            request_body={"TranscriptionJobName": "litellm-job-1"},
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider=None,
        )

        assert normalized["kwargs"].get("model") != "transcribe/GetTranscriptionJob"
