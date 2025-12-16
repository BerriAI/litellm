import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import litellm


def test_convert_chat_completion_messages_to_responses_api_image_input():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()

    user_content = "What's in this image?"
    user_image = "https://w7.pngwing.com/pngs/666/274/png-transparent-image-pictures-icon-photo-thumbnail.png"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_content,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": user_image},
                },
            ],
        },
    ]

    response, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    response_str = json.dumps(response)

    assert user_content in response_str
    assert user_image in response_str

    print("response: ", response)
    assert response[0]["content"][1]["image_url"] == user_image


def test_convert_chat_completion_messages_to_responses_api_tool_result_with_image():
    """
    Test that tool messages with image content are correctly transformed to Responses API format.

    This is a regression test for issue #17762 where images in tool results were not
    correctly transformed from Chat Completion format (image_url with nested object)
    to Responses API format (input_image with flat string).

    Chat Completion format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    Responses API format:
        {"type": "input_image", "image_url": "data:image/png;base64,..."}
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()

    test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    # Chat Completion format with image in tool result
    messages = [
        {
            "role": "user",
            "content": "Fetch the image from this URL",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "fetch_image",
                        "arguments": '{"url": "https://example.com/image.png"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": test_image_base64},
                }
            ],
        },
        {
            "role": "user",
            "content": "What color is the image?",
        },
    ]

    response, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Find the function_call_output item
    function_call_output = None
    for item in response:
        if item.get("type") == "function_call_output":
            function_call_output = item
            break

    assert (
        function_call_output is not None
    ), "function_call_output not found in response"
    assert function_call_output["call_id"] == "call_abc123"

    # Check that the output is correctly transformed
    output = function_call_output["output"]
    assert isinstance(output, list), "output should be a list"
    assert len(output) == 1, "output should have one item"

    image_item = output[0]
    # Should be transformed to Responses API format
    assert (
        image_item["type"] == "input_image"
    ), f"Expected type 'input_image', got '{image_item.get('type')}'"
    assert (
        image_item["image_url"] == test_image_base64
    ), "image_url should be a flat string, not a nested object"
    assert "detail" in image_item, "detail field should be present"

    print("✓ Tool result with image correctly transformed to Responses API format")


def test_openai_responses_chunk_parser_reasoning_summary():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {
        "delta": "**Compar",
        "item_id": "rs_686d544208748198b6912e27b7c299c00e24bd875d35bade",
        "output_index": 0,
        "sequence_number": 4,
        "summary_index": 0,
        "type": "response.reasoning_summary_text.delta",
    }

    result = iterator.chunk_parser(chunk)

    assert isinstance(result, ModelResponseStream)
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert isinstance(choice, StreamingChoices)
    assert choice.index == 0
    delta = choice.delta
    assert isinstance(delta, Delta)
    assert delta.content is None
    assert delta.reasoning_content == "**Compar"
    assert delta.tool_calls is None
    assert delta.function_call is None


def test_chunk_parser_string_output_text_delta_produces_text():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )
    from litellm.types.utils import ModelResponseStream

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {"type": "response.output_text.delta", "delta": "literal text"}

    result = iterator.chunk_parser(chunk)

    assert isinstance(result, ModelResponseStream)
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.delta.content == "literal text"
    assert choice.delta.tool_calls is None
    assert choice.finish_reason is None


def test_chunk_parser_enum_output_text_delta_produces_text():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )
    from litellm.types.llms.openai import ResponsesAPIStreamEvents
    from litellm.types.utils import ModelResponseStream

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {"type": ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, "delta": "enum text"}

    result = iterator.chunk_parser(chunk)

    assert isinstance(result, ModelResponseStream)
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.delta.content == "enum text"
    assert choice.delta.tool_calls is None
    assert choice.finish_reason is None


def test_chunk_parser_function_call_added_produces_tool_use():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )
    from litellm.types.llms.openai import ResponsesAPIStreamEvents
    from litellm.types.utils import ModelResponseStream

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {
        "type": ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        "arguments": '{"key": "value"}',
        "item": {"type": "function_call", "name": "fn", "call_id": "call-42"},
    }

    result = iterator.chunk_parser(chunk)

    assert isinstance(result, ModelResponseStream)
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.delta.tool_calls is not None
    assert len(choice.delta.tool_calls) == 1
    tool_call = choice.delta.tool_calls[0]
    assert tool_call.id == "call-42"
    assert tool_call.type == "function"
    assert tool_call.function.name == "fn"
    assert tool_call.function.arguments == '{"key": "value"}'
    assert choice.finish_reason is None


def test_transform_response_with_reasoning_and_output():
    """Test transform_response handles ResponsesAPIResponse with reasoning items and output messages."""
    from unittest.mock import Mock

    from openai.types.responses import ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_reasoning_item import (
        ResponseReasoningItem,
        Summary,
    )

    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from litellm.types.llms.openai import (
        InputTokensDetails,
        OutputTokensDetails,
        ResponseAPIUsage,
        ResponsesAPIResponse,
    )
    from litellm.types.utils import ModelResponse, Usage

    handler = LiteLLMResponsesTransformationHandler()

    # Create the reasoning item with summary
    reasoning_summary = Summary(
        text="**Creating a poem**\n\nThe user wants a poem without constraints, which is great! I need to focus on keeping it original and evocative.",
        type="summary_text",
    )
    reasoning_item = ResponseReasoningItem(
        id="rs_04c8021b8b3188a00068e9ae08c2d8819d82268b129351a979",
        summary=[reasoning_summary],
        type="reasoning",
        content=None,
        encrypted_content=None,
        status=None,
    )

    # Create the output message with the poem
    poem_text = """I found a pocket of evening
hidden behind the gutters of the day —
a small, folded sky of blue
that hummed like a hush.

The streetlight rehearsed its first apology,
slowly pulling down the curtain
on the city's impatient laughter.
Windows blinked awake like tired eyes,
and the air remembered rain it once promised.

You walked by with a map of quiet in your hands,
tracing routes that led away from all the clocks.
For a moment the coffee shop's bell
tied our minutes together — bright and accidental —
and the world refined itself to the size of that bell's sound.

We did not name the solitude; we sipped it.
You left a warmth on the bench like a small sun,
and night stitched the rest into blue and shadow.
Tomorrow will bring its petitions and promises,
but for now the city breathes slow and wide,
and I learn to carry this small calm home."""

    output_text = ResponseOutputText(
        annotations=[], text=poem_text, type="output_text", logprobs=[]
    )
    output_message = ResponseOutputMessage(
        id="msg_04c8021b8b3188a00068e9ae0b92f4819dac64d85b4abb67ec",
        content=[output_text],
        role="assistant",
        status="completed",
        type="message",
    )

    # Create usage information
    usage = ResponseAPIUsage(
        input_tokens=16,
        input_tokens_details=InputTokensDetails(
            audio_tokens=None, cached_tokens=0, text_tokens=None
        ),
        output_tokens=195,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0, text_tokens=None),
        total_tokens=211,
        cost=None,
    )

    # Create the full ResponsesAPIResponse
    raw_response = ResponsesAPIResponse(
        id="resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfMDRjODAyMWI4YjMxODhhMDAwNjhlOWFlMDgyYmZjODE5ZDhmNDk0OTI5MWMzMzM4YTc=",
        created_at=1760144904,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model="gpt-5-mini-2025-08-07",
        object="response",
        output=[reasoning_item, output_message],
        parallel_tool_calls=True,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning={"effort": "low", "summary": "detailed"},
        status="completed",
        text={"format": {"type": "text"}, "verbosity": "medium"},
        truncation="disabled",
        usage=usage,
        user=None,
        store=True,
        background=False,
        billing={"payer": "developer"},
        max_tool_calls=None,
        prompt_cache_key=None,
        safety_identifier=None,
        service_tier="default",
        top_logprobs=0,
    )

    # Create empty model_response
    model_response = ModelResponse(
        id="chatcmpl-42e863c4-7a31-4229-84f3-4c3a6eeb7610",
        created=1760144904,
        model=None,
        object="chat.completion",
        system_fingerprint=None,
        choices=[],
        usage=Usage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
    )

    # Create mock objects for required parameters
    logging_obj = Mock()
    messages = [{"role": "user", "content": "Think of a poem, and then write it."}]
    request_data = {"model": "gpt-5-mini"}
    optional_params = {"reasoning_effort": "low", "extra_body": {}}
    litellm_params = {"acompletion": False, "api_key": None}
    encoding = Mock()

    # Call transform_response
    result = handler.transform_response(
        model="gpt-5-mini",
        raw_response=raw_response,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data=request_data,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        encoding=encoding,
        api_key=None,
        json_mode=None,
    )

    # Assertions
    assert result.model == "gpt-5-mini"
    assert len(result.choices) == 1

    # Check the choice
    choice = result.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert choice.message.content == poem_text

    # Check usage
    assert result.usage.prompt_tokens == 16
    assert result.usage.completion_tokens == 195
    assert result.usage.total_tokens == 211

    # Check reasoning content
    assert choice.message.reasoning_content == reasoning_summary.text

    print("✓ transform_response correctly handled reasoning items and output messages")


def test_convert_tools_to_responses_format():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()

    tools = [{"type": "function", "function": {"name": "test", "arguments": "test"}}]

    result = handler._convert_tools_to_responses_format(tools)

    assert result[0]["name"] == "test"


def test_extract_extra_body_params_reasoning_effort_override():
    """Test that reasoning_effort from extra_body overrides top-level reasoning_effort"""
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()

    # Test case: reasoning_effort in extra_body (with summary) should override top-level string
    optional_params = {
        "reasoning_effort": "high",  # Top-level string
        "extra_body": {
            "reasoning_effort": {
                "effort": "high",
                "summary": {"type": "summary_text"},
            },  # More complete dict in extra_body
            "previous_response_id": "resp_123",  # Supported param
            "custom_param": "will_stay_in_extra_body",  # Unsupported param
        },
    }

    result = handler._extract_extra_body_params(optional_params)

    # reasoning_effort from extra_body should override top-level
    assert result["reasoning_effort"] == {
        "effort": "high",
        "summary": {"type": "summary_text"},
    }

    # previous_response_id should be extracted to top-level
    assert result["previous_response_id"] == "resp_123"

    # extra_body should no longer be in optional_params (it was popped)
    assert "extra_body" not in result


def test_transform_request_single_char_keys_not_matched():
    """Test that single-character keys are not incorrectly matched to 'metadata' or 'previous_response_id'

    This is a regression test for a bug where:
    - key in ("metadata") was used instead of key == "metadata"
    - In Python, ("metadata") is a string, not a tuple
    - So "m" in ("metadata") returns True (character in string)
    - This caused single-char keys like "m", "e", "t", etc. to incorrectly match
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()

    # Create mock objects
    logging_obj = Mock()

    messages = [{"role": "user", "content": "test"}]

    # Test with single-character keys that are in "metadata" string
    # These should NOT be treated as metadata
    optional_params = {
        "m": "should_not_be_metadata",  # "m" is in "metadata"
        "e": "should_not_be_metadata",  # "e" is in "metadata"
        "t": "should_not_be_metadata",  # "t" is in "metadata"
        "p": "should_not_be_previous_response_id",  # "p" is in "previous_response_id"
        "r": "should_not_be_previous_response_id",  # "r" is in "previous_response_id"
    }

    litellm_params = {}
    headers = {}

    result = handler.transform_request(
        model="gpt-4",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
        litellm_logging_obj=logging_obj,
    )

    # Verify that single-char keys were NOT mapped to metadata or previous_response_id
    assert result.get("metadata") != "should_not_be_metadata"
    assert result.get("previous_response_id") != "should_not_be_previous_response_id"

    # Now test that the actual keys DO work correctly
    optional_params_correct = {
        "metadata": {"user_id": "123"},
        "previous_response_id": "resp_abc",
    }

    result_correct = handler.transform_request(
        model="gpt-4",
        messages=messages,
        optional_params=optional_params_correct,
        litellm_params=litellm_params,
        headers=headers,
        litellm_logging_obj=logging_obj,
    )

    # Verify that the correct keys ARE mapped properly
    assert result_correct.get("metadata") == {"user_id": "123"}
    assert result_correct.get("previous_response_id") == "resp_abc"

    print(
        "✓ Single-character keys are not incorrectly matched to metadata/previous_response_id"
    )


# =============================================================================
# Tests for issue #17246: Streaming tool_calls dropped when text + tool_calls
# =============================================================================


def test_message_done_does_not_emit_is_finished():
    """
    Test that OUTPUT_ITEM_DONE for a message does NOT emit is_finished=True.
    This is the core fix for issue #17246.

    Before fix: message completion emitted is_finished=True, causing tool_calls
    that came after to be dropped.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {
        "type": "response.output_item.done",
        "item": {"type": "message", "content": []},
    }

    result = iterator.chunk_parser(chunk)

    # After the fix, message completion should NOT set finish_reason
    # ModelResponseStream doesn't have is_finished - check finish_reason instead
    assert len(result.choices) > 0, "result should have choices"
    assert (
        result.choices[0].finish_reason is None or result.choices[0].finish_reason == ""
    ), "message completion should not emit finish_reason"


def test_response_completed_emits_is_finished():
    """
    Test that response.completed DOES emit is_finished=True.
    This ensures streaming ends properly after ALL output items are sent.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {"type": "response.completed"}

    result = iterator.chunk_parser(chunk)

    # response.completed should emit finish_reason='stop'
    assert len(result.choices) > 0, "result should have choices"
    assert (
        result.choices[0].finish_reason == "stop"
    ), "response.completed should emit finish_reason='stop'"


def test_function_call_done_emits_is_finished():
    """
    Test that OUTPUT_ITEM_DONE for a function_call still emits is_finished=True.
    This preserves existing behavior for tool_calls.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_123",
            "arguments": '{"location": "Tokyo"}',
        },
    }

    result = iterator.chunk_parser(chunk)

    # function_call completion should emit finish_reason='tool_calls'
    assert len(result.choices) > 0, "result should have choices"
    assert (
        result.choices[0].finish_reason == "tool_calls"
    ), "function_call should emit finish_reason='tool_calls'"
    assert (
        result.choices[0].delta.tool_calls is not None
        and len(result.choices[0].delta.tool_calls) > 0
    ), "function_call should include tool_calls"


def test_text_plus_tool_calls_sequence():
    """
    Test the full sequence when model returns text + tool_calls.
    This is the main scenario for issue #17246.

    Expected: is_finished=True should NOT appear until function_call is done,
    not when message is done.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=None, sync_stream=True
    )

    # Simulate the sequence from OpenAI Responses API
    chunks = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": "!"},
        {
            "type": "response.output_item.done",
            "item": {"type": "message", "content": []},
        },  # message done
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_123",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "delta": '{"location":"Tokyo"}',
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_123",
                "arguments": '{"location":"Tokyo"}',
            },
        },
        {"type": "response.completed"},
    ]

    results = [iterator.chunk_parser(chunk) for chunk in chunks]

    # Check message done (index 2) does NOT have finish_reason set
    message_done_result = results[2]
    assert len(message_done_result.choices) > 0, "message done should have choices"
    assert (
        message_done_result.choices[0].finish_reason is None
        or message_done_result.choices[0].finish_reason == ""
    ), "message done should not have finish_reason"

    # Check function_call done (index 5) DOES have finish_reason='tool_calls'
    function_done_result = results[5]
    assert (
        len(function_done_result.choices) > 0
    ), "function_call done should have choices"
    assert (
        function_done_result.choices[0].finish_reason == "tool_calls"
    ), "function_call done should have finish_reason='tool_calls'"

    # Check response.completed (index 6) has finish_reason='stop'
    completed_result = results[6]
    assert len(completed_result.choices) > 0, "response.completed should have choices"
    assert (
        completed_result.choices[0].finish_reason == "stop"
    ), "response.completed should have finish_reason='stop'"
