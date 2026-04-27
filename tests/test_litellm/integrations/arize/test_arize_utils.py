import json
import os
import sys
from typing import Optional

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio

import pytest

import litellm
from litellm.integrations._types.open_inference import (
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from litellm.integrations.arize.arize import ArizeLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Choices, StandardCallbackDynamicParams


def test_arize_set_attributes():
    """
    Test setting attributes for Arize, including all custom LLM attributes.
    Ensures that the correct span attributes are being added during a request.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import ModelResponse

    span = MagicMock()  # Mocked tracing span to test attribute setting

    # Construct kwargs to simulate a real LLM request scenario
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Basic Request Content"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {"key_1": "value_1", "key_2": None},
            "call_type": "completion",
        },
        "optional_params": {
            "max_tokens": "100",
            "temperature": "1",
            "top_p": "5",
            "stream": False,
            "user": "test_user",
            "tools": [
                {
                    "function": {
                        "name": "get_weather",
                        "description": "Fetches weather details.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["location"],
                        },
                    }
                }
            ],
            "functions": [{"name": "get_weather"}, {"name": "get_stock_price"}],
        },
        "litellm_params": {"custom_llm_provider": "openai"},
    }

    # Simulated LLM response object
    response_obj = ModelResponse(
        usage={"total_tokens": 100, "completion_tokens": 60, "prompt_tokens": 40},
        choices=[
            Choices(message={"role": "assistant", "content": "Basic Response Content"})
        ],
        model="gpt-4o",
        id="chatcmpl-ID",
    )

    # Apply attribute setting via ArizeLogger
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    # Validate that the expected number of attributes were set
    assert span.set_attribute.call_count == 26

    # Metadata attached to the span
    span.set_attribute.assert_any_call(
        SpanAttributes.METADATA, json.dumps({"key_1": "value_1", "key_2": None})
    )

    # Basic LLM information
    span.set_attribute.assert_any_call(SpanAttributes.LLM_MODEL_NAME, "gpt-4o")
    span.set_attribute.assert_any_call("llm.request.type", "completion")
    span.set_attribute.assert_any_call(SpanAttributes.LLM_PROVIDER, "openai")

    # LLM generation parameters
    span.set_attribute.assert_any_call("llm.request.max_tokens", "100")
    span.set_attribute.assert_any_call("llm.request.temperature", "1")
    span.set_attribute.assert_any_call("llm.request.top_p", "5")

    # Streaming and user info
    span.set_attribute.assert_any_call("llm.is_streaming", "False")
    span.set_attribute.assert_any_call("llm.user", "test_user")

    # Response metadata
    span.set_attribute.assert_any_call("llm.response.id", "chatcmpl-ID")
    span.set_attribute.assert_any_call("llm.response.model", "gpt-4o")
    # Span kind is set to TOOL when tools are present
    span.set_attribute.assert_any_call(SpanAttributes.OPENINFERENCE_SPAN_KIND, "TOOL")

    # Request message content and metadata
    span.set_attribute.assert_any_call(
        SpanAttributes.INPUT_VALUE, "Basic Request Content"
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
        "user",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
        "Basic Request Content",
    )

    # Tool call definitions and function names
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_TOOLS}.0.name", "get_weather"
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_TOOLS}.0.description",
        "Fetches weather details.",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_TOOLS}.0.parameters",
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            }
        ),
    )

    # Invocation parameters
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_INVOCATION_PARAMETERS, '{"user": "test_user"}'
    )

    # User ID
    span.set_attribute.assert_any_call(SpanAttributes.USER_ID, "test_user")

    # Output message content
    span.set_attribute.assert_any_call(
        SpanAttributes.OUTPUT_VALUE, "Basic Response Content"
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
        "assistant",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
        "Basic Response Content",
    )

    # Token counts
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 100)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 60)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 40)


def test_arize_set_attributes_responses_api():
    """
    Test setting attributes for Responses API with mixed output (reasoning + message).
    Verifies that multiple output types are correctly handled.
    """
    from unittest.mock import MagicMock
    from litellm.types.llms.openai import (
        ResponsesAPIResponse,
        ResponseAPIUsage,
        OutputTokensDetails,
    )
    from openai.types.responses import (
        ResponseReasoningItem,
        ResponseOutputMessage,
        ResponseOutputText,
    )
    from openai.types.responses.response_reasoning_item import Summary

    span = MagicMock()  # Mocked tracing span to test attribute setting

    # Construct kwargs to simulate a real LLM request scenario
    kwargs = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "What is the answer?"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user", "stream": True},
            "metadata": {"key_1": "value_1", "key_2": None},
            "call_type": "responses",
        },
        "optional_params": {
            "max_tokens": "100",
            "temperature": "1",
            "top_p": "5",
            "stream": True,
            "user": "test_user",
        },
        "litellm_params": {"custom_llm_provider": "openai"},
    }

    # Simulate Responses API response with mixed output
    response_obj = ResponsesAPIResponse(
        id="response-123",
        created_at=1625247600,
        output=[
            ResponseReasoningItem(
                id="reasoning-001",
                type="reasoning",
                summary=[
                    Summary(text="First, I need to analyze...", type="summary_text")
                ],
            ),
            ResponseOutputMessage(
                id="msg-001",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="The answer is 42",
                        type="output_text",
                    )
                ],
            ),
        ],
        usage=ResponseAPIUsage(
            input_tokens=120,
            output_tokens=250,
            total_tokens=370,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=180),
        ),
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    # Verify reasoning summary was set (index 0)
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_REASONING_SUMMARY}",
        "First, I need to analyze...",
    )

    # Verify message content was set (index 1)
    span.set_attribute.assert_any_call(SpanAttributes.OUTPUT_VALUE, "The answer is 42")
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}",
        "The answer is 42",
    )
    span.set_attribute.assert_any_call(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}",
        "assistant",
    )

    # Verify token counts including reasoning tokens
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 370)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 250)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 120)
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, 180
    )


def test_arize_set_attributes_anthropic_cache_tokens():
    """
    Anthropic prompt-caching populates both cache_read and cache_creation;
    LiteLLM normalizes them onto `prompt_tokens_details`, and they should be
    emitted as OpenInference cache_read / cache_write span attributes so
    observability backends can display the cache breakdown and apply correct
    cost calculation (cache reads at 0.1x, cache writes at 1.25x).
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse, Usage

    span = MagicMock()
    kwargs = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {"stream": False},
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }
    response_obj = ModelResponse(
        usage=Usage(
            prompt_tokens=4276,
            completion_tokens=50,
            total_tokens=4326,
            cache_read_input_tokens=4000,
            cache_creation_input_tokens=261,
        ),
        choices=[Choices(message={"role": "assistant", "content": "Hello"})],
        model="claude-sonnet-4",
        id="msg-cache-1",
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 4326)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 50)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 4276)
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, 4000
    )
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, 261
    )


def test_arize_set_attributes_openai_cached_tokens():
    """
    OpenAI's native cache surfaces only cached_tokens (no creation). The
    cache_read attribute must still be emitted; cache_write should be omitted.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import (
        Choices,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {"stream": False},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage=Usage(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=500),
        ),
        choices=[Choices(message={"role": "assistant", "content": "Hello"})],
        model="gpt-4o",
        id="msg-openai-cache",
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, 500
    )
    attribute_keys = [c.args[0] for c in span.set_attribute.call_args_list]
    assert (
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE not in attribute_keys
    )


def test_arize_set_attributes_deepseek_cache_hit_tokens():
    """
    DeepSeek surfaces cache hits via `prompt_cache_hit_tokens`; LiteLLM
    normalizes this onto `prompt_tokens_details.cached_tokens`, which the
    OTEL emitter must pick up.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse, Usage

    span = MagicMock()
    kwargs = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hi"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {"stream": False},
        "litellm_params": {"custom_llm_provider": "deepseek"},
    }
    response_obj = ModelResponse(
        usage=Usage(
            prompt_tokens=2000,
            completion_tokens=100,
            total_tokens=2100,
            prompt_cache_hit_tokens=1500,
        ),
        choices=[Choices(message={"role": "assistant", "content": "Hello"})],
        model="deepseek-chat",
        id="msg-deepseek-cache",
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, 1500
    )


def test_arize_set_attributes_no_cache_tokens_omits_attributes():
    """
    Without cache tokens (no caching, or first-time prompt below threshold),
    cache_read / cache_write attributes must not be emitted.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse, Usage

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {"stream": False},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        choices=[Choices(message={"role": "assistant", "content": "Hi"})],
        model="gpt-4o",
        id="resp-no-cache",
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    attribute_keys = [c.args[0] for c in span.set_attribute.call_args_list]
    assert (
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attribute_keys
    )
    assert (
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE not in attribute_keys
    )


class TestArizeLogger(CustomLogger):
    """
    Custom logger implementation to capture standard_callback_dynamic_params.
    Used to verify that dynamic config keys are being passed to callbacks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_callback_dynamic_params: Optional[
            StandardCallbackDynamicParams
        ] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Capture dynamic params and print them for verification
        print("logged kwargs", json.dumps(kwargs, indent=4, default=str))
        self.standard_callback_dynamic_params = kwargs.get(
            "standard_callback_dynamic_params"
        )


@pytest.mark.asyncio
async def test_arize_dynamic_params():
    """
    Test to ensure that dynamic Arize keys (API key and space key)
    are received inside the callback logger at runtime.
    """
    test_arize_logger = TestArizeLogger()
    litellm.callbacks = [test_arize_logger]

    # Perform a mocked async completion call to trigger logging
    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Basic Request Content"}],
        mock_response="test",
        arize_api_key="test_api_key_dynamic",
        arize_space_key="test_space_key_dynamic",
    )

    # Allow for async propagation
    await asyncio.sleep(2)

    # Assert dynamic parameters were received in the callback
    assert test_arize_logger.standard_callback_dynamic_params is not None
    assert (
        test_arize_logger.standard_callback_dynamic_params.get("arize_api_key")
        == "test_api_key_dynamic"
    )
    assert (
        test_arize_logger.standard_callback_dynamic_params.get("arize_space_key")
        == "test_space_key_dynamic"
    )


def test_construct_dynamic_arize_headers():
    """
    Test the construct_dynamic_arize_headers method with various input scenarios.
    Ensures that dynamic Arize headers are properly constructed from callback parameters.
    """
    from litellm.types.utils import StandardCallbackDynamicParams

    # Test with all parameters present
    dynamic_params_full = StandardCallbackDynamicParams(
        arize_api_key="test_api_key", arize_space_id="test_space_id"
    )
    arize_logger = ArizeLogger()

    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_full)
    expected_headers = {"api_key": "test_api_key", "arize-space-id": "test_space_id"}
    assert headers == expected_headers

    # Test with only space_id
    dynamic_params_space_id_only = StandardCallbackDynamicParams(
        arize_space_id="test_space_id"
    )

    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_space_id_only)
    expected_headers = {"arize-space-id": "test_space_id"}
    assert headers == expected_headers

    # Test with empty parameters dict
    dynamic_params_empty = StandardCallbackDynamicParams()

    headers = arize_logger.construct_dynamic_otel_headers(dynamic_params_empty)
    assert headers == {}

    # test with space key and api key
    dynamic_params_space_key_and_api_key = StandardCallbackDynamicParams(
        arize_space_key="test_space_key", arize_api_key="test_api_key"
    )
    headers = arize_logger.construct_dynamic_otel_headers(
        dynamic_params_space_key_and_api_key
    )
    expected_headers = {"arize-space-id": "test_space_key", "api_key": "test_api_key"}
