import json
import os
from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider

from litellm.integrations._types.open_inference import SpanAttributes as OpenInferenceSpanAttributes
from litellm.integrations.weave.weave_otel import (
    WeaveLLMObsOTELAttributes,
    _set_weave_specific_attributes,
    get_weave_otel_config,
    set_weave_otel_attributes,
)
from litellm.types.integrations.weave_otel import WeaveOtelConfig, WeaveSpanAttributes
from litellm.types.utils import Message, ModelResponse


def test_get_weave_otel_config():
    """Test config creation with required env vars and error cases for missing vars."""
    # Test successful config creation with required environment variables
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
        },
        clear=True,
    ):
        config = get_weave_otel_config()

        assert isinstance(config, WeaveOtelConfig)
        assert config.protocol == "otlp_http"
        assert config.project_id == "test-entity/test-project"
        assert config.otlp_auth_headers is not None
        assert "Authorization=" in config.otlp_auth_headers
        assert "project_id=test-entity/test-project" in config.otlp_auth_headers
        assert config.endpoint == "https://trace.wandb.ai/otel/v1/traces"

        # Verify environment variables were set
        assert (
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
            == "https://trace.wandb.ai/otel/v1/traces"
        )
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == config.otlp_auth_headers

    # Test ValueError when WANDB_API_KEY is missing
    with patch.dict(
        os.environ, {"WANDB_PROJECT_ID": "test-entity/test-project"}, clear=True
    ):
        with pytest.raises(ValueError, match="WANDB_API_KEY must be set"):
            get_weave_otel_config()

    # Test ValueError when WANDB_PROJECT_ID is missing
    with patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"}, clear=True):
        with pytest.raises(ValueError, match="WANDB_PROJECT_ID must be set"):
            get_weave_otel_config()


def test_get_weave_otel_config_with_custom_host():
    """Test config creation with custom WANDB_HOST."""
    # Test with host that already has https://
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
            "WANDB_HOST": "https://custom.wandb.io",
        },
        clear=True,
    ):
        config = get_weave_otel_config()
        assert config.endpoint == "https://custom.wandb.io/otel/v1/traces"
        assert (
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
            == "https://custom.wandb.io/otel/v1/traces"
        )

    # Test with host without http:// or https://
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
            "WANDB_HOST": "custom.wandb.io",
        },
        clear=True,
    ):
        config = get_weave_otel_config()
        assert config.endpoint == "https://custom.wandb.io/otel/v1/traces"

    # Test with host with trailing slash
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
            "WANDB_HOST": "https://custom.wandb.io/",
        },
        clear=True,
    ):
        config = get_weave_otel_config()
        assert config.endpoint == "https://custom.wandb.io/otel/v1/traces"


def test_set_weave_specific_attributes_display_name_from_metadata():
    """Test _set_weave_specific_attributes sets display_name from metadata."""
    mock_span = MagicMock()
    kwargs = {
        "metadata": {"display_name": "custom-display-name"},
        "model": "gpt-4",
    }

    with patch(
        "litellm.integrations.weave.weave_otel.safe_set_attribute"
    ) as mock_safe_set:
        _set_weave_specific_attributes(mock_span, kwargs, None)

        # Should set display_name from metadata
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.DISPLAY_NAME.value, "custom-display-name"
        )


def test_set_weave_specific_attributes_display_name_from_model():
    """Test _set_weave_specific_attributes sets display_name from model when not in metadata."""
    mock_span = MagicMock()
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "metadata": {},
    }

    with patch(
        "litellm.integrations.weave.weave_otel.safe_set_attribute"
    ) as mock_safe_set:
        _set_weave_specific_attributes(mock_span, kwargs, None)

        # Should set display_name from model
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.DISPLAY_NAME.value, "openai__gpt-4o-mini"
        )


def test_set_weave_specific_attributes_thread_id_and_is_turn():
    """Test _set_weave_specific_attributes sets thread_id and is_turn from session_id."""
    mock_span = MagicMock()
    kwargs = {
        "metadata": {"session_id": "session-123"},
    }

    with patch(
        "litellm.integrations.weave.weave_otel.safe_set_attribute"
    ) as mock_safe_set:
        _set_weave_specific_attributes(mock_span, kwargs, None)

        # Should set thread_id and is_turn
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.THREAD_ID.value, "session-123"
        )
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.IS_TURN.value, True
        )


_TRACER: Final = TracerProvider().get_tracer("weave-otel-tests")
WEAVE_MESSAGES: Final = [{"role": "user", "content": "weather in SF?"}]
WEAVE_TOOLS: Final = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
WEAVE_FUNCTIONS: Final = [{"name": "get_time", "parameters": {}}]


@pytest.mark.parametrize(
    ("request_kwargs", "expected_input"),
    [
        (
            {
                "messages": WEAVE_MESSAGES,
                "optional_params": {"tools": WEAVE_TOOLS, "functions": WEAVE_FUNCTIONS, "top_p": 0.5},
            },
            {"messages": WEAVE_MESSAGES, "functions": WEAVE_FUNCTIONS, "tools": WEAVE_TOOLS},
        ),
        ({"messages": WEAVE_MESSAGES, "optional_params": None}, {"messages": WEAVE_MESSAGES}),
    ],
)
def test_weave_input_value_carries_the_request_functions_and_tools(
    request_kwargs: Mapping[str, object], expected_input: Mapping[str, object]
) -> None:
    span: Final = _TRACER.start_span("litellm_request")

    WeaveLLMObsOTELAttributes.set_messages(span, request_kwargs)

    assert json.loads(str(span.attributes[OpenInferenceSpanAttributes.INPUT_VALUE])) == expected_input


def test_weave_output_value_is_the_dumped_model_response() -> None:
    span: Final = _TRACER.start_span("litellm_request")
    response: Final = ModelResponse(
        id="chatcmpl-1",
        model="gpt-4o",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Sunny."}}],
    )

    _set_weave_specific_attributes(span, {"model": "gpt-4o"}, response)

    output: Final = json.loads(str(span.attributes[OpenInferenceSpanAttributes.OUTPUT_VALUE]))
    assert output["id"] == "chatcmpl-1"
    assert output["choices"][0]["message"]["content"] == "Sunny."


NOT_JSON_ENCODABLE: Final = object()
NON_UTF8_BYTES: Final = b"\x80"


@pytest.mark.parametrize(
    ("messages", "expected_messages"),
    [
        (
            (Message(role="user", content="weather in SF?"), Message(role="assistant", content="Sunny.")),
            [
                Message(role="user", content="weather in SF?").model_dump(),
                Message(role="assistant", content="Sunny.").model_dump(),
            ],
        ),
        (
            (Message(role="user", content="weather in SF?", trace_handle=NOT_JSON_ENCODABLE),),
            [{**Message(role="user", content="weather in SF?").model_dump(), "trace_handle": str(NOT_JSON_ENCODABLE)}],
        ),
        (
            (Message(role="user", content="weather in SF?", audio_bytes=NON_UTF8_BYTES),),
            [str(Message(role="user", content="weather in SF?", audio_bytes=NON_UTF8_BYTES))],
        ),
    ],
    ids=["litellm_messages", "litellm_message_with_value_json_cannot_encode", "litellm_message_with_non_utf8_bytes"],
)
def test_weave_span_keeps_input_and_token_usage_when_messages_are_litellm_messages(
    messages: tuple[Message, ...], expected_messages: Sequence[object]
) -> None:
    span: Final = _TRACER.start_span("litellm_request")
    response: Final = ModelResponse(
        model="gpt-4o",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Still sunny."}}],
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    )

    set_weave_otel_attributes(span, {"messages": messages, "standard_logging_object": {}}, response)

    attributes: Final = span.attributes or {}
    assert json.loads(str(attributes[OpenInferenceSpanAttributes.INPUT_VALUE])) == {"messages": expected_messages}
    assert {
        key: attributes.get(key)
        for key in (
            OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_TOTAL,
        )
    } == {
        OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_PROMPT: 7,
        OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 3,
        OpenInferenceSpanAttributes.LLM_TOKEN_COUNT_TOTAL: 10,
    }


def test_weave_span_input_keeps_dict_message_content_exactly_as_sent() -> None:
    span: Final = _TRACER.start_span("litellm_request")
    messages: Final = ({"role": "user", "content": "before\x00after", "metadata": {1: "one"}},)

    set_weave_otel_attributes(span, {"messages": messages, "standard_logging_object": {}}, ModelResponse(model="gpt-4o"))

    assert json.loads(str((span.attributes or {})[OpenInferenceSpanAttributes.INPUT_VALUE])) == {
        "messages": [{"role": "user", "content": "before\x00after", "metadata": {"1": "one"}}]
    }
