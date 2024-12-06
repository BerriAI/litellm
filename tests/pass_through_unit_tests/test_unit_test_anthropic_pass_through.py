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

    result = AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
        httpx_response=mock_httpx_response,
        response_body=mock_response,
        logging_obj=mock_logging_obj,
        url_route="/v1/chat/completions",
        result="success",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
    )

    assert isinstance(result["result"], litellm.ModelResponse)


@pytest.mark.parametrize(
    "metadata_params",
    [{"metadata": {"user_id": "test"}}, {"litellm_metadata": {"user": "test"}}, {}],
)
def test_create_anthropic_response_logging_payload(mock_logging_obj, metadata_params):
    # Test the logging payload creation
    model_response = litellm.ModelResponse()
    model_response.choices = [{"message": {"content": "Test response"}}]

    start_time = datetime.now()
    end_time = datetime.now()

    result = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
        litellm_model_response=model_response,
        model="claude-3-opus-20240229",
        kwargs={
            "litellm_params": {
                "metadata": {
                    "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
                    "user_api_key_user_id": "default_user_id",
                    "user_api_key_team_id": None,
                    "user_api_key_end_user_id": "default_user_id",
                },
                "api_base": "https://api.anthropic.com/v1/messages",
            },
            "call_type": "pass_through_endpoint",
            "litellm_call_id": "5cf924cb-161c-4c1d-a565-31aa71ab50ab",
            "passthrough_logging_payload": {
                "url": "https://api.anthropic.com/v1/messages",
                "request_body": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Open a new Firefox window, navigate to google.com.",
                                }
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "I'll help you open Firefox and navigate to Google. First, let me check the desktop with a screenshot to locate the Firefox icon.",
                                },
                                {
                                    "type": "tool_use",
                                    "id": "toolu_01Tour7YxyXkwhuSP25dQEP7",
                                    "name": "computer",
                                    "input": {"action": "screenshot"},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_01Tour7YxyXkwhuSP25dQEP7",
                                    "content": "",
                                }
                            ],
                        },
                    ],
                    "tools": [
                        {
                            "type": "computer_20241022",
                            "name": "computer",
                            "display_width_px": 1280,
                            "display_height_px": 800,
                        },
                        {"type": "text_editor_20241022", "name": "str_replace_editor"},
                        {"type": "bash_20241022", "name": "bash"},
                    ],
                    "max_tokens": 4096,
                    "model": "claude-3-5-sonnet-20241022",
                    **metadata_params,
                },
                "response_body": {
                    "id": "msg_015uSaCZBvu9gUSkAmZtMfxC",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet-20241022",
                    "content": [
                        {
                            "type": "text",
                            "text": "Now I'll click on the Firefox icon to launch it.",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_01TQsF5p7Pf4LGKyLUDDySVr",
                            "name": "computer",
                            "input": {"action": "mouse_move", "coordinate": [24, 36]},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2202, "output_tokens": 89},
                },
            },
            "response_cost": 0.007941,
            "model": "claude-3-5-sonnet-20241022",
        },
        start_time=start_time,
        end_time=end_time,
        logging_obj=mock_logging_obj,
    )

    assert isinstance(result, dict)
    assert "model" in result
    assert "response_cost" in result
    assert "standard_logging_object" in result
    if metadata_params:
        assert "test" == result["standard_logging_object"]["end_user"]
    else:
        assert "" == result["standard_logging_object"]["end_user"]
