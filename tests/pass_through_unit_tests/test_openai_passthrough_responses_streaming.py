from datetime import datetime
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def test_is_openai_responses_route():
    """Verify is_openai_responses_route works for both full URLs and relative path url_route."""
    assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("/openai_passthrough/v1/responses")
    assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("/v1/responses")
    assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/responses")
    assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/openai_passthrough/v1/responses")
    assert not OpenAIPassthroughLoggingHandler.is_openai_responses_route("/v1/chat/completions")


def test_handle_logging_openai_collected_chunks_responses():
    """Verify streaming passthrough handler for /v1/responses calculates cost and sets call_type='responses'."""
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"custom_llm_provider": "openai", "litellm_params": {}}

    all_chunks = [
        'data: {"type":"response.created","response":{"id":"resp_0c72","object":"response","status":"in_progress"}}',
        'data: {"type":"response.completed","response":{"id":"resp_0c72","object":"response","created_at":1723360000,"output":[],"model":"gpt-4o-mini-2024-07-18","usage":{"input_tokens":14,"output_tokens":2,"total_tokens":16}}}',
    ]

    res = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=MagicMock(),
        url_route="/openai_passthrough/v1/responses",
        request_body={"model": "gpt-4o-mini", "stream": True},
        endpoint_type=EndpointType.OPENAI,
        start_time=datetime.now(),
        all_chunks=all_chunks,
        end_time=datetime.now(),
    )

    assert isinstance(res["result"], ResponsesAPIResponse)
    assert res["kwargs"]["call_type"] == "responses"
    assert logging_obj.model_call_details["call_type"] == "responses"
    assert pytest.approx(res["kwargs"]["response_cost"], 1e-8) == 3.3e-06
