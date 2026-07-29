"""
Test for response_format to text.format conversion in completion -> responses bridge
"""

import pytest
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)


def test_transform_response_format_to_text_format_json_schema():
    """Test conversion of response_format with json_schema to text.format"""
    handler = LiteLLMResponsesTransformationHandler()

    # Chat Completion format
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "person_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    # Convert to Responses API format
    result = handler._transform_response_format_to_text_format(response_format)

    # Verify conversion
    assert result is not None
    assert "format" in result
    assert result["format"]["type"] == "json_schema"
    assert result["format"]["name"] == "person_schema"
    assert result["format"]["strict"] is True
    assert "schema" in result["format"]
    assert result["format"]["schema"]["type"] == "object"
    assert "properties" in result["format"]["schema"]


def test_transform_response_format_to_text_format_json_object():
    """Test conversion of response_format with json_object to text.format"""
    handler = LiteLLMResponsesTransformationHandler()

    response_format = {"type": "json_object"}

    result = handler._transform_response_format_to_text_format(response_format)

    assert result is not None
    assert "format" in result
    assert result["format"]["type"] == "json_object"


def test_transform_response_format_to_text_format_text():
    """Test conversion of response_format with text to text.format"""
    handler = LiteLLMResponsesTransformationHandler()

    response_format = {"type": "text"}

    result = handler._transform_response_format_to_text_format(response_format)

    assert result is not None
    assert "format" in result
    assert result["format"]["type"] == "text"


def test_transform_response_format_to_text_format_none():
    """Test that None input returns None"""
    handler = LiteLLMResponsesTransformationHandler()

    result = handler._transform_response_format_to_text_format(None)

    assert result is None


def test_transform_request_with_response_format():
    """Test that transform_request correctly handles response_format parameter"""
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {"role": "user", "content": "Extract person info: John Doe, 30 years old"}
    ]

    optional_params = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    }

    litellm_params = {}
    headers = {}

    # Mock logging object
    class MockLoggingObj:
        pass

    litellm_logging_obj = MockLoggingObj()

    result = handler.transform_request(
        model="o3-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
        litellm_logging_obj=litellm_logging_obj,
    )

    # Verify that text parameter was set with converted format
    assert "text" in result
    assert result["text"] is not None
    assert "format" in result["text"]
    assert result["text"]["format"]["type"] == "json_schema"
    assert result["text"]["format"]["name"] == "person_schema"
    assert "schema" in result["text"]["format"]


def test_transform_request_includes_extra_headers():
    """Test that transform_request forwards headers as extra_headers for upstream call."""
    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {}
    litellm_params = {}

    class MockLoggingObj:
        pass

    headers = {"cf-aig-authorization": "secret-token"}
    result = handler.transform_request(
        model="gpt-5-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
        litellm_logging_obj=MockLoggingObj(),
    )
    assert result.get("extra_headers") == headers


def test_transform_request_strips_internal_metadata_to_litellm_metadata():
    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {}
    litellm_params = {
        "metadata": {"user_api_key_auth": {"id": "abc"}},
        "litellm_metadata": {"trace_id": "trace-1"},
        "api_key": "sk-test",
    }

    class MockLoggingObj:
        pass

    result = handler.transform_request(
        model="gpt-5-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
        litellm_logging_obj=MockLoggingObj(),
    )

    assert "metadata" not in result
    assert result["litellm_metadata"]["user_api_key_auth"]["id"] == "abc"
    assert result["litellm_metadata"]["trace_id"] == "trace-1"


def test_transform_request_preserves_user_metadata():
    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {"metadata": {"customer_id": "cust-123"}}
    litellm_params = {"metadata": {"internal_key": "secret"}}

    class MockLoggingObj:
        pass

    result = handler.transform_request(
        model="gpt-5-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
        litellm_logging_obj=MockLoggingObj(),
    )

    assert result["metadata"] == {"customer_id": "cust-123"}
    assert "internal_key" not in result["metadata"]
    assert result["litellm_metadata"]["internal_key"] == "secret"


def test_transform_request_drops_user_metadata_with_additional_drop_params():
    from litellm.utils import get_optional_params

    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = get_optional_params(
        model="gpt-4o",
        messages=messages,
        metadata={"customer_id": "cust-123"},
        additional_drop_params=["metadata"],
        custom_llm_provider="openai",
    )
    litellm_params = {"metadata": {"internal_key": "secret"}}

    class MockLoggingObj:
        pass

    result = handler.transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
        litellm_logging_obj=MockLoggingObj(),
    )

    assert "metadata" not in result
    assert result["litellm_metadata"]["internal_key"] == "secret"


def test_translate_responses_chunk_passthrough_chat_completion_chunk():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    chat_chunk = {
        "id": "chatcmpl-cache-passthrough",
        "object": "chat.completion.chunk",
        "created": 1779104834,
        "model": "gpt-5.4",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hi! How can I help?"},
                "finish_reason": None,
            }
        ],
    }

    result = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
        chat_chunk
    )

    assert result.choices[0].delta.content == "Hi! How can I help?"
    assert result.choices[0].finish_reason is None


def test_tool_message_plain_string_passed_through_as_string():
    """
    Plain-string tool output must reach function_call_output.output as a string.

    The Responses API spec documents function_call_output.output as a plain string.
    Strict/enterprise backends reject the list-of-input_text form for simple text.
    Regression guard for the fix to #34978.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    t = LiteLLMResponsesTransformationHandler()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Paris"},
    ]

    items, _ = t.convert_chat_completion_messages_to_responses_api(messages)
    tool_item = next(i for i in items if i.get("type") == "function_call_output")

    assert isinstance(tool_item["output"], str), (
        "Plain-string tool content must pass through as a string, not be wrapped in a list"
    )
    assert tool_item["output"] == "Paris"


def test_tool_message_multimodal_list_still_converted():
    """
    Multimodal (list) tool output must still be converted to Responses API content items.
    This ensures the #17507 fix is preserved for the multimodal case.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    t = LiteLLMResponsesTransformationHandler()
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": [{"type": "text", "text": "result text"}],
        },
    ]

    items, _ = t.convert_chat_completion_messages_to_responses_api(messages)
    tool_item = next(i for i in items if i.get("type") == "function_call_output")

    assert isinstance(tool_item["output"], list), (
        "Multimodal list tool content must be converted to a list of Responses API items"
    )


def test_tool_message_none_content_becomes_empty_string():
    """
    A tool message with content=None must produce an empty string, not an empty
    list. function_call_output.output is a plain string for text results, so the
    empty case has to stay string-typed for strict backends.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    t = LiteLLMResponsesTransformationHandler()
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_3", "content": None},
    ]

    items, _ = t.convert_chat_completion_messages_to_responses_api(messages)
    tool_item = next(i for i in items if i.get("type") == "function_call_output")

    assert tool_item["output"] == ""
    assert isinstance(tool_item["output"], str)


def test_tool_message_non_string_content_is_stringified():
    """
    Unexpected (non-str, non-list) tool content falls back to str(), keeping
    function_call_output.output string-typed rather than wrapping it in a list.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    t = LiteLLMResponsesTransformationHandler()
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_4", "content": 42},
    ]

    items, _ = t.convert_chat_completion_messages_to_responses_api(messages)
    tool_item = next(i for i in items if i.get("type") == "function_call_output")

    assert tool_item["output"] == "42"
    assert isinstance(tool_item["output"], str)
