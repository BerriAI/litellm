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
        self.standard_logging_object = kwargs["standard_logging_object"]
        pass


@pytest.mark.asyncio
async def test_basic_openai_responses_api_streaming_with_logging():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    response = await litellm.aresponses(
        model="gpt-4o",
        input="hi",
        stream=True,
    )

    async for event in response:
        print("litellm response=", json.dumps(event, indent=4, default=str))

    print("sleeping for 2 seconds...")
    await asyncio.sleep(2)
    print(
        "standard logging payload=",
        json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
    )


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

    assert test_custom_logger.standard_logging_object is not None

    # validate token counts match OpenAI response
    assert (
        test_custom_logger.standard_logging_object["prompt_tokens"]
        == response["usage"]["input_tokens"]
    )
    assert (
        test_custom_logger.standard_logging_object["completion_tokens"]
        == response["usage"]["output_tokens"]
    )
    assert (
        test_custom_logger.standard_logging_object["total_tokens"]
        == response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
    )

    # validate spend > 0
    assert test_custom_logger.standard_logging_object["response_cost"] > 0

    # validate response id matches OpenAI
    assert test_custom_logger.standard_logging_object["id"] == response["id"]

    # validate model matches
    assert test_custom_logger.standard_logging_object["model"] == request_model

    # validate messages matches
    assert test_custom_logger.standard_logging_object["messages"] == [
        {"content": "hi", "role": "user"}
    ]

    # validate responses matches
