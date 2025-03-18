import asyncio
import logging

from litellm import Choices
import pytest
from dotenv import load_dotenv

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations._types.open_inference import SpanAttributes
from litellm.integrations.arize.arize import ArizeConfig, ArizeLogger

load_dotenv()


@pytest.mark.asyncio()
async def test_async_otel_callback():
    litellm.set_verbose = True

    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_logger.setLevel(logging.DEBUG)
    litellm.success_callback = ["arize"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from local arize"}],
        mock_response="hello",
        temperature=0.1,
        user="OTEL_USER",
    )

    await asyncio.sleep(2)


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("ARIZE_SPACE_ID", "test_space_id")
    monkeypatch.setenv("ARIZE_API_KEY", "test_api_key")


def test_get_arize_config(mock_env_vars):
    """
    Use Arize default endpoint when no endpoints are provided
    """
    config = ArizeLogger.get_arize_config()
    assert isinstance(config, ArizeConfig)
    assert config.space_id == "test_space_id"
    assert config.api_key == "test_api_key"
    assert config.endpoint == "https://otlp.arize.com/v1"
    assert config.protocol == "otlp_grpc"


def test_get_arize_config_with_endpoints(mock_env_vars, monkeypatch):
    """
    Use provided endpoints when they are set
    """
    monkeypatch.setenv("ARIZE_ENDPOINT", "grpc://test.endpoint")
    monkeypatch.setenv("ARIZE_HTTP_ENDPOINT", "http://test.endpoint")

    config = ArizeLogger.get_arize_config()
    assert config.endpoint == "grpc://test.endpoint"
    assert config.protocol == "otlp_grpc"


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
        "litellm_params": {"metadata": {"key": "value"}},
        "standard_logging_object": {"model_parameters": {"user": "test_user"}}
    }
    response_obj = ModelResponse(usage={"total_tokens": 100, "completion_tokens": 60, "prompt_tokens": 40}, 
                                 choices=[Choices(message={"role": "assistant", "content": "response content"})])

    ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    assert span.set_attribute.call_count == 14
    span.set_attribute.assert_any_call(SpanAttributes.METADATA, str({"key": "value"}))
    span.set_attribute.assert_any_call(SpanAttributes.LLM_MODEL_NAME, "gpt-4o")
    span.set_attribute.assert_any_call(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
    span.set_attribute.assert_any_call(SpanAttributes.INPUT_VALUE, "basic arize test")
    span.set_attribute.assert_any_call("llm.input_messages.0.message.role", "user")
    span.set_attribute.assert_any_call("llm.input_messages.0.message.content", "basic arize test")
    span.set_attribute.assert_any_call(SpanAttributes.LLM_INVOCATION_PARAMETERS, '{"user": "test_user"}')
    span.set_attribute.assert_any_call(SpanAttributes.USER_ID, "test_user")
    span.set_attribute.assert_any_call(SpanAttributes.OUTPUT_VALUE, "response content")
    span.set_attribute.assert_any_call("llm.output_messages.0.message.role", "assistant")
    span.set_attribute.assert_any_call("llm.output_messages.0.message.content", "response content")
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 100)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 60)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 40)