"""
Tests that data_residency is correctly populated on the litellm logging
object's litellm_params for OpenAI Responses paths, even when
custom_llm_provider is resolved from the model string inside responses()
rather than passed explicitly.
"""

import json
from unittest.mock import MagicMock, patch

import litellm


def _make_responses_api_response_body() -> dict:
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 1234567890,
        "model": "gpt-4.1",
        "output": [
            {
                "type": "message",
                "id": "msg-test",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }


def _make_mock_http_client(response_body: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


def _capture_logging_obj():
    captured = {}

    real_init = litellm.Logging.__init__

    def init_spy(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        captured["logging_obj"] = self

    return captured, init_spy


def test_responses_eu_api_base_sets_data_residency():
    """When api_base is a regional OpenAI host and custom_llm_provider is
    inferred from the model (not passed explicitly), data_residency must end
    up on the logging object's litellm_params so the cost calculator can apply
    the regional uplift."""
    mock_client = _make_mock_http_client(_make_responses_api_response_body())
    captured, init_spy = _capture_logging_obj()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=mock_client,
        ),
        patch.object(litellm.Logging, "__init__", init_spy),
    ):
        litellm.responses(
            model="gpt-4.1",
            input="hi",
            api_base="https://eu.api.openai.com/v1",
            api_key="test-key",
        )

    logging_obj = captured["logging_obj"]
    assert logging_obj.litellm_params.get("data_residency") == "eu"


def test_responses_us_api_base_sets_data_residency():
    mock_client = _make_mock_http_client(_make_responses_api_response_body())
    captured, init_spy = _capture_logging_obj()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=mock_client,
        ),
        patch.object(litellm.Logging, "__init__", init_spy),
    ):
        litellm.responses(
            model="gpt-4.1",
            input="hi",
            api_base="https://us.api.openai.com/v1",
            api_key="test-key",
        )

    logging_obj = captured["logging_obj"]
    assert logging_obj.litellm_params.get("data_residency") == "us"


def test_responses_global_api_base_leaves_data_residency_none():
    mock_client = _make_mock_http_client(_make_responses_api_response_body())
    captured, init_spy = _capture_logging_obj()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=mock_client,
        ),
        patch.object(litellm.Logging, "__init__", init_spy),
    ):
        litellm.responses(
            model="gpt-4.1",
            input="hi",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
        )

    logging_obj = captured["logging_obj"]
    assert logging_obj.litellm_params.get("data_residency") is None
