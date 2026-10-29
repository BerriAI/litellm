from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest


from litellm.proxy.pass_through_endpoints.llm_provider_handlers.comprehend_medical_passthrough_logging_handler import (
    ComprehendMedicalPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


def _make_response(operation: str) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "https://comprehendmedical.us-east-1.amazonaws.com/",
        headers={"X-Amz-Target": f"ComprehendMedical_20181030.{operation}"},
    )
    return httpx.Response(200, request=request, text='{"Entities": []}')


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.model_call_details = {}
    return logging_obj


class TestComprehendMedicalCost:
    @pytest.mark.parametrize(
        "operation,text,expected",
        [
            ("DetectEntitiesV2", "x" * 250, 0.03),
            ("DetectEntitiesV2", "x" * 100, 0.01),
            ("DetectPHI", "", 0.0014),
            ("DetectPHI", "x" * 101, 0.0028),
            ("InferICD10CM", "x" * 100, 0.0005),
            ("InferRxNorm", "x" * 150, 0.0005),
            ("InferSNOMEDCT", "x", 0.0075),
            ("StartEntitiesDetectionV2Job", "x" * 1000, 0.0),
        ],
    )
    def test_cost_per_started_100_char_unit(self, operation, text, expected):
        assert ComprehendMedicalPassthroughLoggingHandler.get_cost_for_operation(
            operation=operation, text=text
        ) == pytest.approx(expected)


class TestComprehendMedicalPassthroughHandler:
    def test_records_model_provider_and_cost(self):
        logging_obj = _make_logging_obj()

        handler_result = ComprehendMedicalPassthroughLoggingHandler.comprehend_medical_passthrough_handler(
            httpx_response=_make_response("DetectEntitiesV2"),
            logging_obj=logging_obj,
            url_route="https://comprehendmedical.us-east-1.amazonaws.com/",
            result='{"Entities": []}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"Text": "x" * 250},
        )

        assert handler_result["result"] == {"response": '{"Entities": []}'}
        assert handler_result["kwargs"]["model"] == "comprehendmedical/DetectEntitiesV2"
        assert handler_result["kwargs"]["custom_llm_provider"] == "comprehendmedical"
        assert handler_result["kwargs"]["response_cost"] == pytest.approx(0.03)
        assert "standard_logging_object" in handler_result["kwargs"]
        assert logging_obj.model_call_details["model"] == "comprehendmedical/DetectEntitiesV2"
        assert logging_obj.model_call_details["custom_llm_provider"] == "comprehendmedical"
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.03)

    def test_missing_text_bills_one_unit_minimum(self):
        logging_obj = _make_logging_obj()

        handler_result = ComprehendMedicalPassthroughLoggingHandler.comprehend_medical_passthrough_handler(
            httpx_response=_make_response("DetectPHI"),
            logging_obj=logging_obj,
            url_route="https://comprehendmedical.us-east-1.amazonaws.com/",
            result="{}",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={},
        )

        assert handler_result["kwargs"]["model"] == "comprehendmedical/DetectPHI"
        assert handler_result["kwargs"]["response_cost"] == pytest.approx(0.0014)


class TestIsComprehendMedicalRoute:
    def test_matches_by_provider_tag(self):
        assert PassThroughEndpointLogging().is_comprehend_medical_route("comprehendmedical")

    def test_does_not_match_other_providers(self):
        assert not PassThroughEndpointLogging().is_comprehend_medical_route("bedrock")

    def test_config_driven_passthrough_to_comprehend_host_is_not_claimed(self):
        logging_obj = _make_logging_obj()

        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response("DetectEntitiesV2"),
            response_body={"Entities": []},
            request_body={"Text": "John Smith"},
            logging_obj=logging_obj,
            url_route="https://comprehendmedical.us-east-1.amazonaws.com/",
            result='{"Entities": []}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider=None,
        )

        assert normalized["kwargs"].get("model") != "comprehendmedical/DetectEntitiesV2"
        assert "response_cost" not in normalized["kwargs"]


class TestNormalizeDispatch:
    def test_normalize_routes_to_comprehend_medical_handler(self):
        logging_obj = _make_logging_obj()

        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response("DetectPHI"),
            response_body={"Entities": []},
            request_body={"Text": "John Smith"},
            logging_obj=logging_obj,
            url_route="https://comprehendmedical.us-east-1.amazonaws.com/",
            result='{"Entities": []}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="comprehendmedical",
        )

        assert normalized["standard_logging_response_object"] == {"response": '{"Entities": []}'}
        assert normalized["kwargs"]["model"] == "comprehendmedical/DetectPHI"
        assert normalized["kwargs"]["response_cost"] == pytest.approx(0.0014)
