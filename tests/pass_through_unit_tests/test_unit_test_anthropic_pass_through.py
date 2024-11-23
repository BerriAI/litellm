import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# Import the class we're testing
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)


@pytest.fixture
def mock_response():
    return {
        "model": "claude-3-opus-20240229",
        "content": [{"text": "Hello, world!", "type": "text"}],
        "role": "assistant",
    }


@pytest.fixture
def mock_httpx_response():
    mock_resp = Mock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    return mock_resp


@pytest.fixture
def mock_logging_obj():
    logging_obj = LiteLLMLoggingObj(
        model="claude-3-opus-20240229",
        messages=[],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="123",
        function_id="456",
    )

    logging_obj.async_success_handler = AsyncMock()
    return logging_obj


@pytest.mark.asyncio
async def test_anthropic_passthrough_handler(
    mock_httpx_response, mock_response, mock_logging_obj
):
    """
    Unit test - Assert that the anthropic passthrough handler calls the litellm logging object's async_success_handler
    """
    start_time = datetime.now()
    end_time = datetime.now()

    await AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
        httpx_response=mock_httpx_response,
        response_body=mock_response,
        logging_obj=mock_logging_obj,
        url_route="/v1/chat/completions",
        result="success",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
    )

    # Assert that async_success_handler was called
    assert mock_logging_obj.async_success_handler.called

    call_args = mock_logging_obj.async_success_handler.call_args
    call_kwargs = call_args.kwargs
    print("call_kwargs", call_kwargs)

    # Assert required fields are present in call_kwargs
    assert "result" in call_kwargs
    assert "start_time" in call_kwargs
    assert "end_time" in call_kwargs
    assert "cache_hit" in call_kwargs
    assert "response_cost" in call_kwargs
    assert "model" in call_kwargs
    assert "standard_logging_object" in call_kwargs

    # Assert specific values and types
    assert isinstance(call_kwargs["result"], litellm.ModelResponse)
    assert isinstance(call_kwargs["start_time"], datetime)
    assert isinstance(call_kwargs["end_time"], datetime)
    assert isinstance(call_kwargs["cache_hit"], bool)
    assert isinstance(call_kwargs["response_cost"], float)
    assert call_kwargs["model"] == "claude-3-opus-20240229"
    assert isinstance(call_kwargs["standard_logging_object"], dict)


def test_create_anthropic_response_logging_payload(mock_logging_obj):
    # Test the logging payload creation
    model_response = litellm.ModelResponse()
    model_response.choices = [{"message": {"content": "Test response"}}]

    start_time = datetime.now()
    end_time = datetime.now()

    result = (
        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=model_response,
            model="claude-3-opus-20240229",
            kwargs={},
            start_time=start_time,
            end_time=end_time,
            logging_obj=mock_logging_obj,
        )
    )

    assert isinstance(result, dict)
    assert "model" in result
    assert "response_cost" in result
    assert "standard_logging_object" in result
