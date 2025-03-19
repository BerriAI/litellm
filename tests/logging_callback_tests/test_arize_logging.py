import os
import sys
import time
from unittest.mock import Mock, patch
import json
from litellm.main import completion
import opentelemetry.exporter.otlp.proto.grpc.trace_exporter

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from litellm.integrations._types.open_inference import SpanAttributes
from litellm.integrations.arize.arize import ArizeConfig, ArizeLogger
import litellm
from litellm.types.utils import Choices


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
