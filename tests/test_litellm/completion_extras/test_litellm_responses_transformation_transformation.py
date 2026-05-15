"""
Test for response_format to text.format conversion in completion -> responses bridge
"""

import pytest
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
    OpenAiResponsesToChatCompletionStreamIterator,
)


def test_transform_response_format_to_text_format_json_schema():
    """Test conversion of response_format with json_schema to text.format"""
    handler = LiteLLMResponsesTransformationHandler()

    # Chat Completion format
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "person_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    # Convert to Responses API format
    result = handler._transform_response_format_to_text_format(response_format)

    # Verify conversion
    assert result is not None
    assert "format" in result
    assert result["format"]["type"] == "json_schema"
    assert result["format"]["name"] == "person_schema"
    assert result["format"]["strict"] is True
    assert "schema" in result["format"]
    assert result["format"]["schema"]["type"] == "object"
    assert "properties" in result["format"]["schema"]


def test_transform_response_format_to_text_format_json_object():
    """Test conversion of response_format with json_object to text.format"""
    handler = LiteLLMResponsesTransformationHandler()

    response_format = {"type": "json_object"}

    result = handler._transform_response_format_to_text_format(response_format)

    assert result is not None
    assert "format" in result
    assert result["format"]["type"] == "json_object"


def test_transform_response_format_to_text_format_text():
    """Test conversion of response_format with text to text.format"""
    handler = LiteLLMResponsesTransformationHandler()

    response_format = {"type": "text"}

    result = handler._transform_response_format_to_text_format(response_format)

    assert result is not None
    assert "format" in result
    assert result["format"]["type"] == "text"


def test_transform_response_format_to_text_format_none():
    """Test that None input returns None"""
    handler = LiteLLMResponsesTransformationHandler()

    result = handler._transform_response_format_to_text_format(None)

    assert result is None


def test_transform_request_with_response_format():
    """Test that transform_request correctly handles response_format parameter"""
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {"role": "user", "content": "Extract person info: John Doe, 30 years old"}
    ]

    optional_params = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    }

    litellm_params = {}
    headers = {}

    # Mock logging object
    class MockLoggingObj:
        pass

    litellm_logging_obj = MockLoggingObj()

    result = handler.transform_request(
        model="o3-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
        litellm_logging_obj=litellm_logging_obj,
    )

    # Verify that text parameter was set with converted format
    assert "text" in result
    assert result["text"] is not None
    assert "format" in result["text"]
    assert result["text"]["format"]["type"] == "json_schema"
    assert result["text"]["format"]["name"] == "person_schema"
    assert "schema" in result["text"]["format"]


def test_transform_request_includes_extra_headers():
    """Test that transform_request forwards headers as extra_headers for upstream call."""
    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {}
    litellm_params = {}

    class MockLoggingObj:
        pass

    headers = {"cf-aig-authorization": "secret-token"}
    result = handler.transform_request(
        model="gpt-5-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
        litellm_logging_obj=MockLoggingObj(),
    )
    assert result.get("extra_headers") == headers


def test_transform_request_strips_internal_metadata_to_litellm_metadata():
    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {}
    litellm_params = {
        "metadata": {"user_api_key_auth": {"id": "abc"}},
        "litellm_metadata": {"trace_id": "trace-1"},
        "api_key": "sk-test",
    }

    class MockLoggingObj:
        pass

    result = handler.transform_request(
        model="gpt-5-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
        litellm_logging_obj=MockLoggingObj(),
    )

    assert "metadata" not in result
    assert result["litellm_metadata"]["user_api_key_auth"]["id"] == "abc"
    assert result["litellm_metadata"]["trace_id"] == "trace-1"


def test_transform_request_preserves_user_metadata():
    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {"metadata": {"customer_id": "cust-123"}}
    litellm_params = {"metadata": {"internal_key": "secret"}}

    class MockLoggingObj:
        pass

    result = handler.transform_request(
        model="gpt-5-pro",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
        litellm_logging_obj=MockLoggingObj(),
    )

    assert result["metadata"] == {"customer_id": "cust-123"}
    assert "internal_key" not in result["metadata"]
    assert result["litellm_metadata"]["internal_key"] == "secret"


def test_transform_request_drops_user_metadata_with_additional_drop_params():
    from litellm.utils import get_optional_params

    handler = LiteLLMResponsesTransformationHandler()
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = get_optional_params(
        model="gpt-4o",
        messages=messages,
        metadata={"customer_id": "cust-123"},
        additional_drop_params=["metadata"],
        custom_llm_provider="openai",
    )
    litellm_params = {"metadata": {"internal_key": "secret"}}

    class MockLoggingObj:
        pass

    result = handler.transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
        litellm_logging_obj=MockLoggingObj(),
    )

    assert "metadata" not in result
    assert result["litellm_metadata"]["internal_key"] == "secret"


# ---------------------------------------------------------------------------
# Tests for OpenAiResponsesToChatCompletionStreamIterator.chunk_parser
# (specifically the no-delta / .done-only tool-call path fixed in #27797)
# ---------------------------------------------------------------------------


def _make_iterator() -> OpenAiResponsesToChatCompletionStreamIterator:
    """Return a fresh iterator with an empty (no-op) underlying stream."""
    return OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter([]),
        sync_stream=True,
    )


def test_chunk_parser_done_without_delta_emits_arguments():
    """Models that never send .delta (e.g. gpt-5.3-codex-spark) deliver the
    full function arguments only in the .done event.  chunk_parser must emit a
    tool-call chunk carrying those arguments rather than silently dropping them
    (which would produce arguments='{}' in the reconstructed response)."""
    iterator = _make_iterator()
    arguments_json = '{"city": "Paris"}'

    done_chunk = {
        "type": "response.function_call_arguments.done",
        "output_index": 0,
        "arguments": arguments_json,
    }

    result = iterator.chunk_parser(done_chunk)

    assert result.choices is not None and len(result.choices) == 1
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].function.arguments == arguments_json


def test_chunk_parser_done_after_delta_is_skipped():
    """When .delta chunks were already streamed for an output_index, the .done
    event must NOT re-emit arguments (they are already accumulated by the
    standard delta path, and duplicating them would corrupt the response)."""
    iterator = _make_iterator()
    output_index = 2

    # Simulate receiving a delta first
    delta_chunk = {
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "delta": '{"city":',
    }
    iterator.chunk_parser(delta_chunk)  # records output_index in _seen_arg_delta_idxs

    # Now send the corresponding .done — it should delegate to the static
    # translator, which returns a non-tool-call chunk (no tool_calls in delta).
    done_chunk = {
        "type": "response.function_call_arguments.done",
        "output_index": output_index,
        "arguments": '{"city": "Paris"}',
    }
    result = iterator.chunk_parser(done_chunk)

    # The static translator does not handle .done → no tool_calls emitted
    delta = result.choices[0].delta
    assert not delta.tool_calls


def test_chunk_parser_done_without_delta_uses_output_index():
    """The tool-call chunk emitted for the .done-only path must carry the
    correct output_index so the caller can reassemble multi-tool responses."""
    iterator = _make_iterator()
    output_index = 3

    done_chunk = {
        "type": "response.function_call_arguments.done",
        "output_index": output_index,
        "arguments": "{}",
    }

    result = iterator.chunk_parser(done_chunk)
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].index == output_index


def test_chunk_parser_seen_delta_idxs_tracked_per_index():
    """Delta tracking must be per output_index: a delta for index 0 must not
    suppress the .done emission for index 1."""
    iterator = _make_iterator()

    # Delta only for index 0
    iterator.chunk_parser(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "x",
        }
    )

    # .done for index 1 (no delta seen → should emit)
    arguments_json = '{"q": 42}'
    result = iterator.chunk_parser(
        {
            "type": "response.function_call_arguments.done",
            "output_index": 1,
            "arguments": arguments_json,
        }
    )

    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].function.arguments == arguments_json
    assert tool_calls[0].index == 1
