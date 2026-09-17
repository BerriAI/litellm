from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.typesafe_passthrough_logging_handler import (
    TypeSafePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


def _response() -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone"),
        json={"model": "jev-1.13.0"},
    )


def _logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return logging_obj


def _handler_result(response_body: dict, request_body: dict) -> dict:
    return TypeSafePassthroughLoggingHandler.typesafe_passthrough_handler(
        httpx_response=_response(),
        response_body=response_body,
        logging_obj=_logging_obj(),
        url_route="https://api.typesafe.ai/v1/systemone",
        result='{"answers": {}}',
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body=request_body,
    )


def test_uses_registry_pricing_and_standard_usage():
    logging_obj = _logging_obj()
    model_key = "typesafe/jev-1.13.0"
    model_cost = litellm.model_cost[model_key]
    response = TypeSafePassthroughLoggingHandler.typesafe_passthrough_handler(
        httpx_response=_response(),
        response_body={"model": "jev-1.13.0", "usage": {"input_tokens": 312, "output_tokens": 48}},
        logging_obj=logging_obj,
        url_route="https://api.typesafe.ai/v1/systemone",
        result='{"answers": {}}',
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"model": "jev-latest"},
    )

    expected_cost = 312 * model_cost["input_cost_per_token"] + 48 * model_cost["output_cost_per_token"]
    assert response["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert response["kwargs"]["combined_usage_object"].prompt_tokens == 312
    assert response["kwargs"]["combined_usage_object"].completion_tokens == 48
    assert response["kwargs"]["combined_usage_object"].total_tokens == 360


def test_falls_back_to_request_model_when_response_model_is_missing():
    result = _handler_result(
        {"usage": {"input_tokens": 10, "output_tokens": 2}},
        {"model": "jev-latest"},
    )

    model_cost = litellm.model_cost["typesafe/jev-latest"]
    expected_cost = 10 * model_cost["input_cost_per_token"] + 2 * model_cost["output_cost_per_token"]
    assert result["kwargs"]["model"] == "typesafe/jev-latest"
    assert result["kwargs"]["response_cost"] == pytest.approx(expected_cost)


def test_missing_usage_is_zero_cost():
    result = _handler_result({"model": "jev-1.13.0"}, {"model": "jev-latest"})

    assert result["kwargs"]["response_cost"] == 0.0


def test_records_model_provider_and_cost_on_logging_details():
    logging_obj = _logging_obj()
    result = TypeSafePassthroughLoggingHandler.typesafe_passthrough_handler(
        httpx_response=_response(),
        response_body={"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 0}},
        logging_obj=logging_obj,
        url_route="https://api.typesafe.ai/v1/systemone",
        result="{}",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"model": "jev-latest"},
    )

    assert result["kwargs"]["model"] == "typesafe/jev-1.13.0"
    assert result["kwargs"]["custom_llm_provider"] == "typesafe"
    assert result["kwargs"]["response_cost"] > 0
    assert logging_obj.model_call_details["model"] == "typesafe/jev-1.13.0"
    assert logging_obj.model_call_details["custom_llm_provider"] == "typesafe"
    assert logging_obj.model_call_details["response_cost"] == result["kwargs"]["response_cost"]


def test_success_handler_dispatches_to_typesafe_handler():
    logging_obj = _logging_obj()
    normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=_response(),
        response_body={"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 0}},
        request_body={"model": "jev-latest"},
        logging_obj=logging_obj,
        url_route="https://api.typesafe.ai/v1/systemone",
        result="{}",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        custom_llm_provider="typesafe",
    )

    assert normalized["kwargs"]["custom_llm_provider"] == "typesafe"
    assert normalized["kwargs"]["model"] == "typesafe/jev-1.13.0"
