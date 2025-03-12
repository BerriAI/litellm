import os
import sys
import pytest
import asyncio
from typing import Optional

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse


@pytest.mark.asyncio
async def test_basic_openai_responses_api():
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="gpt-4o", input="Tell me a three sentence bedtime story about a unicorn."
    )
    print("litellm response=", json.dumps(response, indent=4, default=str))

    # validate_responses_api_response()


@pytest.mark.asyncio
async def test_basic_openai_responses_api_streaming():
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="gpt-4o",
        input="Tell me a three sentence bedtime story about a unicorn.",
        stream=True,
    )

    async for event in response:
        print("litellm response=", json.dumps(event, indent=4, default=str))


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
    print("response=", json.dumps(response, indent=4, default=str))
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


@pytest.mark.asyncio
async def test_basic_openai_responses_api_streaming_with_logging():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    request_model = "gpt-4o"
    response = await litellm.aresponses(
        model=request_model,
        input="hi",
        stream=True,
    )
    final_response: Optional[ResponseCompletedEvent] = None
    async for event in response:
        if event.type == "response.completed":
            final_response = event
        print("litellm response=", json.dumps(event, indent=4, default=str))

    print("sleeping for 2 seconds...")
    await asyncio.sleep(2)
    print(
        "standard logging payload=",
        json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
    )

    assert final_response is not None
    assert test_custom_logger.standard_logging_object is not None

    validate_standard_logging_payload(
        slp=test_custom_logger.standard_logging_object,
        response=final_response.response,
        request_model=request_model,
    )


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
        slp_response["usage"]["input_tokens"]
        == litellm_response["usage"]["input_tokens"]
    ), "Input tokens mismatch"
    assert (
        slp_response["usage"]["output_tokens"]
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


@pytest.mark.asyncio
async def test_basic_openai_responses_api_non_streaming_with_logging():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    request_model = "gpt-4o"
    response = await litellm.aresponses(
        model=request_model,
        input="hi",
    )

    print("litellm response=", json.dumps(response, indent=4, default=str))
    print("response hidden params=", response._hidden_params)

    print("sleeping for 2 seconds...")
    await asyncio.sleep(2)
    print(
        "standard logging payload=",
        json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
    )

    assert response is not None
    assert test_custom_logger.standard_logging_object is not None

    validate_standard_logging_payload(
        test_custom_logger.standard_logging_object, response, request_model
    )
