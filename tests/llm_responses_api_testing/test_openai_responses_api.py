import asyncio
import json
import os
import time
from typing import Optional, cast

import pytest
from base_responses_api import BaseResponsesAPITest, validate_responses_api_response

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
)
from litellm.types.utils import StandardLoggingPayload


class TestOpenAIResponsesAPITest(BaseResponsesAPITest):
    test_responses_api_with_tool_calls = None

    def get_base_completion_call_args(self):
        return {
            "model": "openai/gpt-5.5",
        }

    def get_advanced_model_for_shell_tool(self):
        return "openai/gpt-5.2"


class TestCustomLogger(CustomLogger):
    def __init__(
        self,
    ):
        self.standard_logging_object: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async_log_success_event")
        print("kwargs=", json.dumps(kwargs, indent=4, default=str))
        self.standard_logging_object = kwargs["standard_logging_object"]
        pass


def validate_standard_logging_payload(
    slp: StandardLoggingPayload, response: ResponsesAPIResponse, request_model: str
):
    """
    Validate that a StandardLoggingPayload object matches the expected response

    Args:
        slp (StandardLoggingPayload): The standard logging payload object to validate
        response (dict): The litellm response to compare against
        request_model (str): The model name that was requested
    """
    # Validate payload exists
    assert slp is not None, "Standard logging payload should not be None"

    # Validate token counts
    print(
        "VALIDATING STANDARD LOGGING PAYLOAD. response=",
        json.dumps(response, indent=4, default=str),
    )
    print("FIELDS IN SLP=", json.dumps(slp, indent=4, default=str))
    print("SLP PROMPT TOKENS=", slp["prompt_tokens"])
    print("RESPONSE PROMPT TOKENS=", response["usage"]["input_tokens"])
    assert (
        slp["prompt_tokens"] == response["usage"]["input_tokens"]
    ), "Prompt tokens mismatch"
    assert (
        slp["completion_tokens"] == response["usage"]["output_tokens"]
    ), "Completion tokens mismatch"
    assert (
        slp["total_tokens"]
        == response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
    ), "Total tokens mismatch"

    # Validate spend and response metadata
    assert slp["response_cost"] > 0, "Response cost should be greater than 0"
    assert slp["id"] == response["id"], "Response ID mismatch"
    assert slp["model"] == request_model, "Model name mismatch"

    # Validate messages
    assert slp["messages"] == [{"content": "hi", "role": "user"}], "Messages mismatch"

    # Validate complete response structure
    validate_responses_match(slp["response"], response)


def validate_responses_match(slp_response, litellm_response):
    """Validate that the standard logging payload OpenAI response matches the litellm response"""
    # Validate core fields
    assert slp_response["id"] == litellm_response["id"], "ID mismatch"
    assert slp_response["model"] == litellm_response["model"], "Model mismatch"
    assert (
        slp_response["created_at"] == litellm_response["created_at"]
    ), "Created at mismatch"

    # Validate usage
    assert (
        slp_response["usage"]["prompt_tokens"]
        == litellm_response["usage"]["input_tokens"]
    ), "Input tokens mismatch"
    assert (
        slp_response["usage"]["completion_tokens"]
        == litellm_response["usage"]["output_tokens"]
    ), "Output tokens mismatch"
    assert (
        slp_response["usage"]["total_tokens"]
        == litellm_response["usage"]["total_tokens"]
    ), "Total tokens mismatch"

    # Validate output/messages
    assert len(slp_response["output"]) == len(
        litellm_response["output"]
    ), "Output length mismatch"
    for slp_msg, litellm_msg in zip(slp_response["output"], litellm_response["output"]):
        assert slp_msg["role"] == litellm_msg.role, "Message role mismatch"
        # Access the content's text field for the litellm response
        litellm_content = litellm_msg.content[0].text if litellm_msg.content else ""
        assert (
            slp_msg["content"][0]["text"] == litellm_content
        ), f"Message content mismatch. Expected {litellm_content}, Got {slp_msg['content']}"
        assert slp_msg["status"] == litellm_msg.status, "Message status mismatch"


def validate_stream_event(event):
    """
    Validate that a streaming event from litellm.responses() or litellm.aresponses()
    with stream=True conforms to the expected structure based on its event type.

    Args:
        event: The streaming event object to validate

    Raises:
        AssertionError: If the event doesn't match the expected structure for its type
    """
    # Common validation for all event types
    assert hasattr(event, "type"), "Event should have a 'type' attribute"

    # Type-specific validation
    if event.type == "response.created" or event.type == "response.in_progress":
        assert hasattr(
            event, "response"
        ), f"{event.type} event should have a 'response' attribute"
        validate_responses_api_response(event.response, final_chunk=False)

    elif event.type == "response.completed":
        assert hasattr(
            event, "response"
        ), "response.completed event should have a 'response' attribute"
        validate_responses_api_response(event.response, final_chunk=True)
        # Usage is guaranteed only on the completed event
        assert (
            "usage" in event.response
        ), "response.completed event should have usage information"
        print("Usage in event.response=", event.response["usage"])
        assert isinstance(event.response["usage"], ResponseAPIUsage)
    elif event.type == "response.failed" or event.type == "response.incomplete":
        assert hasattr(
            event, "response"
        ), f"{event.type} event should have a 'response' attribute"

    elif (
        event.type == "response.output_item.added"
        or event.type == "response.output_item.done"
    ):
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "item"
        ), f"{event.type} event should have an 'item' attribute"

    elif (
        event.type == "response.content_part.added"
        or event.type == "response.content_part.done"
    ):
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "part"
        ), f"{event.type} event should have a 'part' attribute"

    elif event.type == "response.output_text.delta":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "delta"
        ), f"{event.type} event should have a 'delta' attribute"

    elif event.type == "response.output_text.annotation.added":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "annotation_index"
        ), f"{event.type} event should have an 'annotation_index' attribute"
        assert hasattr(
            event, "annotation"
        ), f"{event.type} event should have an 'annotation' attribute"

    elif event.type == "response.output_text.done":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "text"
        ), f"{event.type} event should have a 'text' attribute"

    elif event.type == "response.refusal.delta":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "delta"
        ), f"{event.type} event should have a 'delta' attribute"

    elif event.type == "response.refusal.done":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "content_index"
        ), f"{event.type} event should have a 'content_index' attribute"
        assert hasattr(
            event, "refusal"
        ), f"{event.type} event should have a 'refusal' attribute"

    elif event.type == "response.function_call_arguments.delta":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "delta"
        ), f"{event.type} event should have a 'delta' attribute"

    elif event.type == "response.function_call_arguments.done":
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "arguments"
        ), f"{event.type} event should have an 'arguments' attribute"

    elif event.type in [
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ]:
        assert hasattr(
            event, "output_index"
        ), f"{event.type} event should have an 'output_index' attribute"
        assert hasattr(
            event, "item_id"
        ), f"{event.type} event should have an 'item_id' attribute"

    elif event.type == "error":
        assert hasattr(
            event, "message"
        ), "Error event should have a 'message' attribute"
    return True  # Return True if validation passes


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_compact_responses_api(sync_mode):
    """
    Test the compact_responses API for OpenAI.

    This test verifies that the compact_responses endpoint works correctly
    for compressing conversation history.
    """
    litellm.turn_on_debug()
    litellm.set_verbose = True

    input_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "What is the weather like today?"},
    ]

    try:
        if sync_mode:
            response = litellm.compact_responses(
                model="openai/gpt-5.5",
                input=input_messages,
                instructions="Be helpful and concise",
            )
        else:
            response = await litellm.acompact_responses(
                model="openai/gpt-5.5",
                input=input_messages,
                instructions="Be helpful and concise",
            )
    except litellm.InternalServerError:
        pytest.skip("Skipping test due to InternalServerError")
    except litellm.BadRequestError as e:
        # compact_responses may not be available for all models/accounts
        pytest.skip(f"Skipping test due to BadRequestError: {e}")

    print("compact_responses response=", json.dumps(response, indent=4, default=str))

    # Validate response structure
    assert response is not None
    assert "id" in response, "Response should have an 'id' field"
    assert "output" in response, "Response should have an 'output' field"
    assert isinstance(response["output"], list), "Output should be a list"
