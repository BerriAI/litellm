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
        "model": "claude-sonnet-4-5-20250929",
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
                    "user_api_key_end_user_id": ("test" if metadata_params else ""),
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
                    "model": "claude-sonnet-4-5-20250929",
                    **metadata_params,
                },
                "response_body": {
                    "id": "msg_015uSaCZBvu9gUSkAmZtMfxC",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
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
            "model": "claude-sonnet-4-5-20250929",
        },
        start_time=start_time,
        end_time=end_time,
        logging_obj=mock_logging_obj,
    )

    assert isinstance(result, dict)
    assert "model" in result
    assert "response_cost" in result


@pytest.mark.parametrize(
    "end_user_id",
    [{"litellm_metadata": {"user": "test"}}, {"metadata": {"user_id": "test"}}],
)
def test_get_user_from_metadata(end_user_id):
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
        PassthroughStandardLoggingPayload,
    )

    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="https://api.anthropic.com/v1/messages",
        request_body={**end_user_id},
        response_body={
            "id": "msg_015uSaCZBvu9gUSkAmZtMfxC",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
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
    )

    response = AnthropicPassthroughLoggingHandler._get_user_from_metadata(
        passthrough_logging_payload=passthrough_logging_payload
    )

    assert response == "test"


@pytest.fixture
def all_chunks():
    return [
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"msg_01G7T4YSBzHjmgTyizv1UfkB","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":17,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":5}}}',
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        "event: ping",
        'data: {"type": "ping"}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here are 5 "}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"important events from the 19th century ("}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1801-1900):\\n\\n1. The Industrial"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Revolution (ongoing throughout the century)\\nMajor technological"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" advancements and societal changes as manufacturing shifted from han"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"d production to machines and factories.\\n\\n2. American Civil War (1861"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"-1865)\\nA conflict between the Union and the"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Confederacy over issues including slavery, resulting in the preservation of the"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" United States and the abolition of slavery.\\n\\n3. Publication"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" of Charles Darwin\'s \\"On the Origin of Species\\" ("}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1859)\\nDarwin\'s groundbreaking work"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" on evolution by natural selection revolutionized biology an"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"d scientific thought.\\n\\n4. Unification of Germany"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" (1871)\\nThe consolidation of numerous"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" German states into a single nation-state under Prussian"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" leadership, led by Otto von Bismarck"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":".\\n\\n5. Abolition of Slavery in Various"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Countries\\nIncluding the British Empire (1833),"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" French colonies (1848), and the United States ("}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1865), marking significant progress in human rights."}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\\n\\nThese events had far-reaching consequences that shape"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"d the modern world in various ways, from politics and economics to science an"}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"d social structures."}}',
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":249}}',
        "event: message_stop",
        'data: {"type":"message_stop"}',
    ]


def test_handle_logging_anthropic_collected_chunks(all_chunks):
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
        PassthroughStandardLoggingPayload,
        EndpointType,
    )
    from litellm.types.utils import ModelResponse

    litellm_logging_obj = Mock()
    pass_through_logging_obj = Mock()

    sent_args = {
        "litellm_logging_obj": litellm_logging_obj,
        "passthrough_success_handler_obj": pass_through_logging_obj,
        "url_route": "https://api.anthropic.com/v1/messages",
        "request_body": {
            "model": "claude-3-5-sonnet-20240620",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "List 5 important events in the XIX century",
                        }
                    ],
                }
            ],
            "max_tokens": 4096,
            "stream": True,
        },
        "endpoint_type": "anthropic",
        "start_time": "2025-01-15T16:04:46.155054",
        "end_time": "2025-01-15T16:04:49.603348",
        "all_chunks": all_chunks,
    }

    result = (
        AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            **sent_args
        )
    )

    assert isinstance(result["result"], ModelResponse)
    print("result=", json.dumps(result, indent=4, default=str))


def test_build_complete_streaming_response(all_chunks):
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )
    from litellm.types.utils import ModelResponse

    litellm_logging_obj = Mock()

    result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=all_chunks,
        model="claude-3-5-sonnet-20240620",
        litellm_logging_obj=litellm_logging_obj,
    )

    assert isinstance(result, ModelResponse)
    assert result.usage.prompt_tokens == 17
    assert result.usage.completion_tokens == 249
    assert result.usage.total_tokens == 266
