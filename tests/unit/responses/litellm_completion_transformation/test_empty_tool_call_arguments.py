import json
from unittest.mock import AsyncMock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    TOOL_CALLS_CACHE,
    LiteLLMCompletionResponsesConfig,
)


def _get_tool_call_arguments(tool_call: object) -> str:
    """Helper to extract arguments string from dict or object tool call chunk."""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            return fn.get("arguments", "")
        return getattr(fn, "arguments", "")
    fn = getattr(tool_call, "function", None)
    if isinstance(fn, dict):
        return fn.get("arguments", "")
    return getattr(fn, "arguments", "")


class TestEmptyToolCallArgumentsSerialization:
    """Issue #40369: Default empty tool call arguments to "{}" strictly at terminal serialization points."""

    def test_transform_function_call_empty_arguments_produces_valid_json(self):
        """Downstream JSON parser requires valid JSON; empty arguments string must deserialize."""
        function_call = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_empty_args_1",
            "arguments": "",
        }
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call
        )
        assert len(messages) == 1
        tool_call = messages[0]["tool_calls"][0]
        arguments_str = _get_tool_call_arguments(tool_call)
        # Must be valid JSON string "{}" so downstream JSON parsers do not throw BadRequestError
        assert arguments_str == "{}"
        assert json.loads(arguments_str) == {}

    def test_transform_function_call_none_arguments_produces_valid_json(self):
        """None or omitted arguments must default to valid JSON "{}"."""
        function_call = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_none_args_1",
            "arguments": None,
        }
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call
        )
        tool_call = messages[0]["tool_calls"][0]
        arguments_str = _get_tool_call_arguments(tool_call)
        assert arguments_str == "{}"
        assert json.loads(arguments_str) == {}

    def test_transform_function_call_omitted_arguments_produces_valid_json(self):
        """Omitted arguments key must default to valid JSON "{}"."""
        function_call = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_omitted_args_1",
        }
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            function_call
        )
        tool_call = messages[0]["tool_calls"][0]
        arguments_str = _get_tool_call_arguments(tool_call)
        assert arguments_str == "{}"
        assert json.loads(arguments_str) == {}

    def test_transform_custom_tool_call_empty_input_produces_valid_content_json(self):
        """Empty custom tool input must serialize as {"content": ""} rather than empty string or invalid schema."""
        custom_call = {
            "type": "custom_tool_call",
            "name": "patch_editor",
            "call_id": "call_custom_empty_1",
            "input": "",
        }
        messages = LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
            custom_call
        )
        tool_call = messages[0]["tool_calls"][0]
        arguments_str = _get_tool_call_arguments(tool_call)
        assert json.loads(arguments_str) == {"content": ""}

    def test_transform_tool_call_output_reconstructed_tool_call_empty_args(self):
        """Cached tool use definition with empty arguments must default to "{}"."""
        call_id = "call_tool_out_empty_args"
        TOOL_CALLS_CACHE.set_cache(
            key=call_id,
            value={
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "fetch_data",
                    "arguments": "",
                },
            },
        )
        output_item = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": "sample output",
        }
        messages = (
            LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
                output_item
            )
        )
        # First message is assistant with replayed tool_calls, second is tool output
        assert len(messages) == 2
        assistant_msg = messages[0]
        replayed_call = assistant_msg["tool_calls"][0]
        arguments_str = _get_tool_call_arguments(replayed_call)
        assert arguments_str == "{}"
        assert json.loads(arguments_str) == {}

    def test_transform_responses_output_tool_call_empty_args(self):
        """Output tool call with empty arguments must serialize to "{}"."""
        tool_call_dict = {
            "id": "call_out_empty_args_1",
            "name": "lookup",
            "arguments": "",
        }
        result = (
            LiteLLMCompletionResponsesConfig._transform_responses_output_tool_call_to_chat_completion_output_tool_call(
                tool_call_dict
            )
        )
        assert result.function.arguments == "{}"
        assert json.loads(result.function.arguments) == {}

    def test_streaming_accumulators_preserve_multi_chunk_argument_buffers(self):
        """Intermediate streaming accumulators must not wipe buffer with truthy "{}" when chunks arrive."""
        iterator = LiteLLMCompletionStreamingIterator(
            model="test-model",
            litellm_custom_stream_wrapper=AsyncMock(),
            request_input="Test input",
            responses_api_request={},
        )

        call_id = "call_stream_accum_1"

        # Chunk 1: Tool call opens, arguments are empty string
        iterator._queue_tool_call_delta_events(
            [{"id": call_id, "type": "function", "function": {"name": "search_query", "arguments": ""}}]
        )

        # Chunk 2: First argument delta
        iterator._queue_tool_call_delta_events(
            [{"id": call_id, "type": "function", "function": {"arguments": '{"query": '}}]
        )

        # Chunk 3: Second argument delta
        iterator._queue_tool_call_delta_events(
            [{"id": call_id, "type": "function", "function": {"arguments": '"litellm"}'}}]
        )

        # The accumulated arguments buffer in streaming_iterator must match the concatenated JSON,
        # not corrupted by any truthy "{}"
        assert iterator._tool_args_by_call_id[call_id] == '{"query": "litellm"}'
        assert json.loads(iterator._tool_args_by_call_id[call_id]) == {"query": "litellm"}
