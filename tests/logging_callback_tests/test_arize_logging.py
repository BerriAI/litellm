import os
import sys
import time
from unittest.mock import Mock, patch
import json
import opentelemetry.exporter.otlp.proto.grpc.trace_exporter
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from litellm.integrations._types.open_inference import SpanAttributes
from litellm.integrations.arize.arize import ArizeConfig, ArizeLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.main import completion
import litellm
from litellm.types.utils import Choices, StandardCallbackDynamicParams
import pytest
import asyncio


def test_arize_set_attributes():
    """
    Test setting attributes for Arize
    """
    from unittest.mock import MagicMock
    from litellm.types.utils import ModelResponse

    span = MagicMock()
    kwargs = {
        "role": "user",
        "content": "simple arize test",
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "basic arize test"}],
        "standard_logging_object": {
            "model_parameters": {"user": "test_user"},
            "metadata": {"key": "value", "key2": None},
        },
    }
    response_obj = ModelResponse(
        usage={"total_tokens": 100, "completion_tokens": 60, "prompt_tokens": 40},
        choices=[Choices(message={"role": "assistant", "content": "response content"})],
    )

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    assert span.set_attribute.call_count == 14
    span.set_attribute.assert_any_call(
        SpanAttributes.METADATA, json.dumps({"key": "value", "key2": None})
    )
    span.set_attribute.assert_any_call(SpanAttributes.LLM_MODEL_NAME, "gpt-4o")
    span.set_attribute.assert_any_call(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
    span.set_attribute.assert_any_call(SpanAttributes.INPUT_VALUE, "basic arize test")
    span.set_attribute.assert_any_call("llm.input_messages.0.message.role", "user")
    span.set_attribute.assert_any_call(
        "llm.input_messages.0.message.content", "basic arize test"
    )
    span.set_attribute.assert_any_call(
        SpanAttributes.LLM_INVOCATION_PARAMETERS, '{"user": "test_user"}'
    )
    span.set_attribute.assert_any_call(SpanAttributes.USER_ID, "test_user")
    span.set_attribute.assert_any_call(SpanAttributes.OUTPUT_VALUE, "response content")
    span.set_attribute.assert_any_call(
        "llm.output_messages.0.message.role", "assistant"
    )
    span.set_attribute.assert_any_call(
        "llm.output_messages.0.message.content", "response content"
    )
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 100)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 60)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 40)


class TestArizeLogger(CustomLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_callback_dynamic_params: Optional[
            StandardCallbackDynamicParams
        ] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("logged kwargs", json.dumps(kwargs, indent=4, default=str))
        self.standard_callback_dynamic_params = kwargs.get(
            "standard_callback_dynamic_params"
        )


@pytest.mark.asyncio
async def test_arize_dynamic_params():
    """verify arize ai dynamic params are recieved by a callback"""
    test_arize_logger = TestArizeLogger()
    litellm.callbacks = [test_arize_logger]
    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "basic arize test"}],
        mock_response="test",
        arize_api_key="test_api_key_dynamic",
        arize_space_key="test_space_key_dynamic",
    )

    await asyncio.sleep(2)

    assert test_arize_logger.standard_callback_dynamic_params is not None
    assert (
        test_arize_logger.standard_callback_dynamic_params.get("arize_api_key")
        == "test_api_key_dynamic"
    )
    assert (
        test_arize_logger.standard_callback_dynamic_params.get("arize_space_key")
        == "test_space_key_dynamic"
    )
