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

    # Validate that the expected number of attributes were set.
    # OPENINFERENCE_SPAN_KIND is written exactly once (defensively, before
    # the main attribute pipeline) so a partial failure cannot blank it.
    # Per the OpenInference spec, a chat completion that passes `tools=[...]`
    # is still an LLM span — not TOOL (TOOL is reserved for actual tool
    # execution by application code).
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
    # Span kind stays LLM even when tools are passed (OpenInference spec).
    span.set_attribute.assert_any_call(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
    # And TOOL must never be written for an LLM chat completion call.
    span_kind_writes = [
        c.args[1]
        for c in span.set_attribute.call_args_list
        if c.args[0] == SpanAttributes.OPENINFERENCE_SPAN_KIND
    ]
    assert "TOOL" not in span_kind_writes

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


def test_set_usage_outputs_pydantic_completion_usage():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/13672

    `_set_usage_outputs` previously called `usage.get(...)` which crashes when
    `usage` is a plain Pydantic model (e.g. openai.types.completion_usage.CompletionUsage)
    that does not implement dict-style `.get()`. Same crash for nested
    `output_tokens_details` / `completion_tokens_details`.

    The function must:
    1. Read total/prompt/completion tokens from a Pydantic usage without `.get`.
    2. Read reasoning_tokens from `completion_tokens_details` (chat completions API)
       OR `output_tokens_details` (responses API), even when those nested objects
       are Pydantic models without `.get`.
    3. Not raise AttributeError; not call span.record_exception.
    """
    from unittest.mock import MagicMock

    from openai.types.completion_usage import (
        CompletionTokensDetails,
        CompletionUsage,
    )

    from litellm.integrations.arize._utils import _set_usage_outputs

    span = MagicMock()

    # Plain OpenAI Pydantic model — has no `.get()`
    usage = CompletionUsage(
        completion_tokens=60,
        prompt_tokens=40,
        total_tokens=100,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=25),
    )
    assert not hasattr(usage, "get"), "precondition: CompletionUsage must lack .get"

    response_obj = {"usage": usage}

    # Must not raise
    _set_usage_outputs(span, response_obj, SpanAttributes)

    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 100)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 40)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 60)
    # reasoning_tokens for chat completions live in completion_tokens_details
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, 25
    )


def test_set_usage_outputs_pydantic_response_api_usage():
    """
    Same crash also affects Responses API with `output_tokens_details` as a
    Pydantic model that lacks `.get()`. Verifies the responses-API path.
    """
    from unittest.mock import MagicMock

    from litellm.integrations.arize._utils import _set_usage_outputs
    from litellm.types.llms.openai import OutputTokensDetails

    # Build an object that mimics openai ResponsesAPI usage but lacks `.get`
    # (uses a plain class — not BaseLiteLLMOpenAIResponseObject)
    class PlainResponsesUsage:
        def __init__(self):
            self.total_tokens = 370
            self.input_tokens = 120
            self.output_tokens = 250
            self.output_tokens_details = OutputTokensDetails(reasoning_tokens=180)

    usage = PlainResponsesUsage()
    assert not hasattr(usage, "get")

    span = MagicMock()
    response_obj = {"usage": usage}

    _set_usage_outputs(span, response_obj, SpanAttributes)

    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 370)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 120)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 250)
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, 180
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


# ---------------------------------------------------------------------------
# Additive rendering-enhancement tests. None of these assert that previously
# emitted attributes were removed or changed — they only assert that the new
# attributes appear in their respective scenarios.
# ---------------------------------------------------------------------------


def _collect_calls(span):
    """Helper: return dict[attr_name] = value of all set_attribute calls."""
    out = {}
    for call in span.set_attribute.call_args_list:
        args = call.args
        if len(args) >= 2:
            out[args[0]] = args[1]
    return out


def test_arize_emits_cache_tokens_openai_style():
    """OpenAI prompt_tokens_details.cached_tokens → cache_read attr."""
    from unittest.mock import MagicMock

    from litellm.integrations.arize._utils import _set_usage_outputs

    span = MagicMock()
    response_obj = {
        "usage": {
            "total_tokens": 100,
            "completion_tokens": 60,
            "prompt_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 32, "audio_tokens": 8},
        }
    }
    _set_usage_outputs(span, response_obj, SpanAttributes)
    attrs = _collect_calls(span)
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 32
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO] == 8


def test_arize_emits_cache_tokens_anthropic_style():
    """Anthropic/Bedrock cache_read_input_tokens / cache_creation_input_tokens."""
    from unittest.mock import MagicMock

    from litellm.integrations.arize._utils import _set_usage_outputs

    span = MagicMock()
    response_obj = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        }
    }
    _set_usage_outputs(span, response_obj, SpanAttributes)
    attrs = _collect_calls(span)
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 80
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] == 20


def test_arize_emits_no_cache_tokens_when_absent():
    """Regression guard: when no cache fields exist, no cache attrs emitted."""
    from unittest.mock import MagicMock

    from litellm.integrations.arize._utils import _set_usage_outputs

    span = MagicMock()
    response_obj = {
        "usage": {"total_tokens": 10, "completion_tokens": 4, "prompt_tokens": 6}
    }
    _set_usage_outputs(span, response_obj, SpanAttributes)
    attrs = _collect_calls(span)
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE not in attrs


def test_passthrough_call_type_resolves_to_llm_span_kind():
    """`allm_passthrough_route` should map to LLM (was UNKNOWN before fix)."""
    from litellm.integrations._types.open_inference import OpenInferenceSpanKindValues
    from litellm.integrations.arize._utils import _infer_open_inference_span_kind

    assert (
        _infer_open_inference_span_kind("allm_passthrough_route")
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert (
        _infer_open_inference_span_kind("llm_passthrough_route")
        == OpenInferenceSpanKindValues.LLM.value
    )


def test_arize_chat_completion_with_tools_stays_llm_span_kind():
    """Regression guard against the old `TOOL` override: a chat completion
    that passes `tools=[...]` AND returns `tool_calls` must remain LLM."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "weather?"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        },
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 10, "completion_tokens": 4, "prompt_tokens": 6},
        choices=[
            Choices(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                }
            )
        ],
        model="gpt-4o",
        id="r-toolkind",
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    span_kind_writes = [
        c.args[1]
        for c in span.set_attribute.call_args_list
        if c.args[0] == SpanAttributes.OPENINFERENCE_SPAN_KIND
    ]
    assert span_kind_writes, "span.kind must be written"
    assert all(v == "LLM" for v in span_kind_writes)
    assert "TOOL" not in span_kind_writes


def test_arize_emits_assistant_tool_calls_on_output_message():
    """Assistant tool_calls should surface as MESSAGE_TOOL_CALLS.* attrs."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "weather?"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 10, "completion_tokens": 4, "prompt_tokens": 6},
        choices=[
            Choices(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                }
            )
        ],
        model="gpt-4o",
        id="chatcmpl-1",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)
    base = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{base}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call_abc"
    assert (
        attrs[f"{base}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
    )
    assert (
        attrs[f"{base}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        == '{"location": "SF"}'
    )


def test_arize_output_value_falls_back_to_tool_calls_summary():
    """When the assistant returns no text content but did request tool
    calls, OUTPUT_VALUE should contain a JSON summary so Arize's Output
    pane shows something."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "weather?"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 10, "completion_tokens": 4, "prompt_tokens": 6},
        choices=[
            Choices(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                }
            )
        ],
        model="gpt-4o",
        id="r-tc-out",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)

    # OUTPUT_VALUE should contain the tool_call name + arguments JSON
    out = attrs[SpanAttributes.OUTPUT_VALUE]
    assert "tool_calls" in out
    assert "get_weather" in out
    assert "SF" in out


def test_arize_output_value_unchanged_when_content_present():
    """Regression guard: when content is non-empty, OUTPUT_VALUE must be
    exactly that content (no summary written)."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 4, "completion_tokens": 2, "prompt_tokens": 2},
        choices=[
            Choices(
                message={
                    "role": "assistant",
                    "content": "hello world",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "n", "arguments": "{}"},
                        }
                    ],
                }
            )
        ],
        model="gpt-4o",
        id="r-content",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)
    assert attrs[SpanAttributes.OUTPUT_VALUE] == "hello world"


def test_arize_emits_tool_call_id_and_name_on_input_tool_message():
    """A tool-result input message should expose tool_call_id + name."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "get_weather",
                "content": "sunny, 72F",
            },
        ],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 10, "completion_tokens": 4, "prompt_tokens": 6},
        choices=[Choices(message={"role": "assistant", "content": "It's sunny."})],
        model="gpt-4o",
        id="chatcmpl-2",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)
    # Assistant tool_call surfaces on input msg index 1
    assistant_base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{assistant_base}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call_abc"
    # Tool message at index 2
    tool_prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.2"
    assert (
        attrs[f"{tool_prefix}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}"] == "call_abc"
    )
    assert attrs[f"{tool_prefix}.{MessageAttributes.MESSAGE_NAME}"] == "get_weather"


def test_arize_emits_multimodal_input_contents():
    """List-shaped content should populate MESSAGE_CONTENTS.* alongside the
    legacy MESSAGE_CONTENT (which stays for back-compat)."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 10, "completion_tokens": 4, "prompt_tokens": 6},
        choices=[Choices(message={"role": "assistant", "content": "A cat."})],
        model="gpt-4o",
        id="chatcmpl-img",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)
    base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attrs[f"{base}.0.message_content.type"] == "text"
    assert attrs[f"{base}.0.message_content.text"] == "What is in this image?"
    assert attrs[f"{base}.1.message_content.type"] == "image"
    assert (
        attrs[f"{base}.1.message_content.image.image.url"]
        == "https://example.com/cat.png"
    )


def test_arize_emits_session_and_user_attrs_from_metadata():
    """end_user_id → SESSION_ID; user_api_key_user_id → USER_ID (only when
    optional_params.user/model_params.user absent)."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {
                "user_api_key_user_id": "user_42",
                "user_api_key_end_user_id": "session_99",
                "user_api_key_team_id": "team_7",
                "user_api_key_team_alias": "alpha",
                "user_api_key_alias": "key_alpha",
            },
            "call_type": "completion",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 4, "completion_tokens": 2, "prompt_tokens": 2},
        choices=[Choices(message={"role": "assistant", "content": "hello"})],
        model="gpt-4o",
        id="r1",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)
    assert attrs[SpanAttributes.SESSION_ID] == "session_99"
    assert attrs[SpanAttributes.USER_ID] == "user_42"
    assert attrs["litellm.team_id"] == "team_7"
    assert attrs["litellm.team_alias"] == "alpha"
    assert attrs["litellm.key_alias"] == "key_alpha"


def test_arize_does_not_use_trace_id_as_session_id_fallback():
    """SESSION_ID must NOT fall back to trace_id (one session-per-request
    would distort Arize Session analytics). trace_id is emitted under its
    own `litellm.trace_id` key instead.
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
            "trace_id": "trace-xyz-123",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 4, "completion_tokens": 2, "prompt_tokens": 2},
        choices=[Choices(message={"role": "assistant", "content": "hi"})],
        model="gpt-4o",
        id="r-trace",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)

    # SESSION_ID must NOT be derived from trace_id.
    assert SpanAttributes.SESSION_ID not in attrs
    # trace_id surfaces under its own key.
    assert attrs["litellm.trace_id"] == "trace-xyz-123"


def test_arize_does_not_overwrite_user_id_from_optional_params():
    """If optional_params.user is set, metadata USER_ID must NOT overwrite."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "standard_logging_object": {
            "model_parameters": {"user": "from_model_params"},
            "metadata": {"user_api_key_user_id": "from_metadata"},
            "call_type": "completion",
        },
        "optional_params": {"user": "from_optional_params"},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 4, "completion_tokens": 2, "prompt_tokens": 2},
        choices=[Choices(message={"role": "assistant", "content": "hello"})],
        model="gpt-4o",
        id="r2",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    user_id_writes = [
        c.args[1]
        for c in span.set_attribute.call_args_list
        if c.args[0] == SpanAttributes.USER_ID
    ]
    assert "from_metadata" not in user_id_writes


def test_arize_emits_response_cost():
    """StandardLoggingPayload.response_cost → llm.cost.total (+ legacy llm.response.cost)."""
    from unittest.mock import MagicMock

    from litellm.types.utils import Choices, ModelResponse

    span = MagicMock()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "completion",
            "response_cost": 0.0012345,
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "openai"},
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 4, "completion_tokens": 2, "prompt_tokens": 2},
        choices=[Choices(message={"role": "assistant", "content": "hello"})],
        model="gpt-4o",
        id="r3",
    )
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)
    assert attrs["llm.cost.total"] == 0.0012345
    assert attrs["llm.response.cost"] == 0.0012345  # legacy key still emitted


def test_arize_passthrough_bedrock_anthropic_normalization():
    """Bedrock-Anthropic passthrough: input/output text must be set so the
    span renders something other than raw provider attrs."""
    from unittest.mock import MagicMock

    span = MagicMock()
    bedrock_response_body = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "The capital of France is Paris."}],
        "model": "anthropic.claude-sonnet-4-v1:0",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 18, "output_tokens": 12},
    }

    class FakeHttpxResponse:
        """Minimal httpx.Response stand-in: has `.text` and no `.get`."""

        def __init__(self, body):
            self.text = json.dumps(body)

    response_obj = FakeHttpxResponse(bedrock_response_body)
    kwargs = {
        "model": "anthropic.claude-sonnet-4-v1:0",
        "messages": [
            {
                "role": "user",
                "content": json.dumps({"messages": [{"role": "user", "content": "?"}]}),
            }
        ],
        "additional_args": {
            "complete_input_dict": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
            }
        },
        "standard_logging_object": {
            "model_parameters": {},
            "metadata": {},
            "call_type": "allm_passthrough_route",
        },
        "optional_params": {},
        "litellm_params": {"custom_llm_provider": "bedrock"},
    }
    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)
    attrs = _collect_calls(span)

    # Input rendering
    assert attrs[SpanAttributes.INPUT_VALUE] == "What is the capital of France?"
    msg0 = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attrs[f"{msg0}.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    assert (
        attrs[f"{msg0}.{MessageAttributes.MESSAGE_CONTENT}"]
        == "What is the capital of France?"
    )

    # Output rendering (Anthropic content[].text)
    assert attrs[SpanAttributes.OUTPUT_VALUE] == "The capital of France is Paris."
    out0 = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attrs[f"{out0}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert (
        attrs[f"{out0}.{MessageAttributes.MESSAGE_CONTENT}"]
        == "The capital of France is Paris."
    )

    # Token counts (Bedrock input_tokens/output_tokens) — extracted via
    # coercion of the non-dict response.
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 18
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 12

    # Span kind defended even though the call_type is a passthrough variant.
    span_kind_writes = [
        c.args[1]
        for c in span.set_attribute.call_args_list
        if c.args[0] == SpanAttributes.OPENINFERENCE_SPAN_KIND
    ]
    assert span_kind_writes  # at least one
    assert all(v == "LLM" for v in span_kind_writes)


def test_arize_passthrough_call_type_does_not_run_on_chat_completion():
    """Guard: passthrough normalizer must not fire for normal chat calls.

    If it did, it could double-write input/output for ordinary completions.
    """
    from unittest.mock import MagicMock

    from litellm.integrations.arize._utils import _maybe_normalize_passthrough

    span = MagicMock()
    _maybe_normalize_passthrough(
        span,
        {
            "additional_args": {
                "complete_input_dict": {"messages": [{"role": "user", "content": "x"}]}
            }
        },
        {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
        {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
        {"call_type": "completion"},
    )
    assert span.set_attribute.call_count == 0


def test_arize_passthrough_skipped_when_message_redaction_enabled():
    """Security guard: when message-logging redaction is enabled, the
    passthrough normalizer must NOT export the raw prompt (read from
    `complete_input_dict`, which bypasses central redaction) to the span.
    """
    from unittest.mock import MagicMock

    from litellm.integrations.arize._utils import _maybe_normalize_passthrough

    span = MagicMock()
    kwargs = {
        "additional_args": {
            "complete_input_dict": {
                "messages": [
                    {"role": "user", "content": "Patient John Doe, SSN 123-45-6789"}
                ]
            }
        },
        # Enables redaction via the dynamic-param path inside
        # should_redact_message_logging(), without touching globals.
        "standard_callback_dynamic_params": {"turn_off_message_logging": True},
    }
    _maybe_normalize_passthrough(
        span,
        kwargs,
        {"content": [{"type": "text", "text": "secret response"}]},
        {"content": [{"type": "text", "text": "secret response"}]},
        {"call_type": "allm_passthrough_route"},
    )
    # Nothing — neither input nor output — should be written to the span.
    assert span.set_attribute.call_count == 0


def test_arize_coerce_response_obj_passes_dicts_through_untouched():
    """Regression guard for the BaseModel/dict path."""
    from litellm.integrations.arize._utils import _coerce_response_obj_for_attrs

    d = {"id": "x", "model": "m"}
    assert _coerce_response_obj_for_attrs(d) is d

    class HasGet:
        def get(self, *a, **k):  # noqa: D401
            return None

    obj = HasGet()
    assert _coerce_response_obj_for_attrs(obj) is obj

    assert _coerce_response_obj_for_attrs(None) is None


def test_arize_coerce_response_obj_parses_httpx_like():
    """httpx.Response-like objects without `.get` should JSON-decode."""
    from litellm.integrations.arize._utils import _coerce_response_obj_for_attrs

    class FakeHttpxResponse:
        text = '{"id": "msg_1", "model": "claude"}'

    parsed = _coerce_response_obj_for_attrs(FakeHttpxResponse())
    assert parsed == {"id": "msg_1", "model": "claude"}


def test_arize_coerce_response_obj_returns_original_on_bad_json():
    from litellm.integrations.arize._utils import _coerce_response_obj_for_attrs

    class BadJson:
        text = "not-json"

    obj = BadJson()
    assert _coerce_response_obj_for_attrs(obj) is obj
