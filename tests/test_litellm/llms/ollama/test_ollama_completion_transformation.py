import json
from unittest.mock import MagicMock, patch

import pytest

from litellm._uuid import uuid
from litellm.litellm_core_utils.prompt_templates.factory import function_call_prompt
from litellm.llms.ollama.completion.transformation import (
    OllamaConfig,
    OllamaTextCompletionResponseIterator,
)
from litellm.types.utils import Message, ModelResponse, ModelResponseStream


class TestOllamaConfig:
    def test_transform_response_standard(self):
        # Initialize config
        config = OllamaConfig()

        # Create mock response
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "Hello, I am an AI assistant",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Return dummy token IDs

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify response
        assert result.choices[0]["message"].content == "Hello, I am an AI assistant"
        assert result.choices[0]["finish_reason"] == "stop"
        assert result.model == "ollama/llama2"
        assert result.created is not None
        # Access usage properly
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    @patch("uuid.uuid4")
    def test_transform_response_json_function_call(self, mock_uuid4):
        # Setup mock UUID
        mock_uuid4.return_value = "test-uuid"

        # Initialize config
        config = OllamaConfig()

        # Create mock response with JSON function call format
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": json.dumps(
                {"name": "get_weather", "arguments": {"location": "San Francisco"}}
            )
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Return dummy token IDs

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify result has tool_calls
        assert result.choices[0]["message"].content is None
        assert result.choices[0]["finish_reason"] == "tool_calls"
        assert len(result.choices[0]["message"].tool_calls) == 1
        assert result.choices[0]["message"].tool_calls[0]["id"].startswith("call_")
        assert (
            result.choices[0]["message"].tool_calls[0]["function"]["name"]
            == "get_weather"
        )
        assert json.loads(
            result.choices[0]["message"].tool_calls[0]["function"]["arguments"]
        ) == {"location": "San Francisco"}
        # No usage assertions here as we don't need to test them in every case

    def test_transform_response_regular_json(self):
        # Initialize config
        config = OllamaConfig()

        # Create mock response with regular JSON (not function call)
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": json.dumps(
                {"result": "success", "data": {"temperature": 72, "unit": "F"}}
            )
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Return dummy token IDs

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify result has JSON content
        expected_content = json.dumps(
            {"result": "success", "data": {"temperature": 72, "unit": "F"}}
        )
        assert result.choices[0]["message"].content == expected_content
        assert result.choices[0]["finish_reason"] == "stop"
        # No usage assertions here as we don't need to test them in every case

    def test_transform_response_with_thinking_tags(self):
        """Test that responses with <think>...</think> tags parse reasoning content correctly."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with thinking tags
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>I need to think about this problem step by step</think>Here is my answer",
            "prompt_eval_count": 15,
            "eval_count": 8,
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted
        assert (
            result.choices[0]["message"].reasoning_content
            == "I need to think about this problem step by step"
        )
        assert result.choices[0]["message"].content == "Here is my answer"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_thinking_tags_alternative(self):
        """Test that responses with <thinking>...</thinking> tags parse reasoning content correctly."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with thinking tags (alternative format)
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<thinking>Let me analyze this carefully</thinking>The solution is X",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted
        assert (
            result.choices[0]["message"].reasoning_content
            == "Let me analyze this carefully"
        )
        assert result.choices[0]["message"].content == "The solution is X"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_multiline_thinking_tags(self):
        """Test that responses with multiline thinking content work correctly."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with multiline thinking content
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>\nThis is a complex problem.\nI need to break it down:\n1. First step\n2. Second step\n</think>Based on my analysis, the answer is Y",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify multiline reasoning content is extracted
        expected_reasoning = "\nThis is a complex problem.\nI need to break it down:\n1. First step\n2. Second step\n"
        assert result.choices[0]["message"].reasoning_content == expected_reasoning
        assert (
            result.choices[0]["message"].content
            == "Based on my analysis, the answer is Y"
        )
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_thinking_only(self):
        """Test response with only thinking content and no additional content."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with only thinking content
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>Just internal thoughts, no response</think>",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted and content is empty
        assert (
            result.choices[0]["message"].reasoning_content
            == "Just internal thoughts, no response"
        )
        assert result.choices[0]["message"].content == ""
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_json_mode_with_thinking_tags(self):
        """Test JSON mode with thinking tags - should handle as text when JSON parsing fails."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with thinking tags in JSON mode
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>Planning my JSON response</think>This is not valid JSON",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted even in JSON mode when JSON parsing fails
        assert (
            result.choices[0]["message"].reasoning_content
            == "Planning my JSON response"
        )
        assert result.choices[0]["message"].content == "This is not valid JSON"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_no_thinking_tags(self):
        """Test that responses without thinking tags work normally."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response without thinking tags
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "Regular response without any thinking tags",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify no reasoning content is extracted
        assert result.choices[0]["message"].reasoning_content is None
        assert (
            result.choices[0]["message"].content
            == "Regular response without any thinking tags"
        )
        assert result.choices[0]["finish_reason"] == "stop"


class TestOllamaTextCompletionResponseIterator:
    def test_chunk_parser_with_thinking_field(self):
        """Test that chunks with 'thinking' field and empty 'response' are handled correctly."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test chunk with thinking field - this is the problematic case from the issue
        chunk_with_thinking = {
            "model": "gpt-oss:20b",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "",
            "thinking": "User",
            "done": False,
        }

        result = iterator.chunk_parser(chunk_with_thinking)

        # Should return a ModelResponseStream with reasoning content
        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert getattr(result.choices[0].delta, "reasoning_content") == "User"

    def test_chunk_parser_normal_response(self):
        """Test that normal response chunks still work."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test normal chunk with response
        normal_chunk = {
            "model": "llama2",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "Hello world",
            "done": False,
        }

        result = iterator.chunk_parser(normal_chunk)

        # Updated to handle ModelResponseStream return type
        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert result.choices[0].delta.content == "Hello world"
        assert getattr(result.choices[0].delta, "reasoning_content", None) is None

    def test_chunk_parser_empty_response_without_thinking(self):
        """Test that empty response chunks without thinking still work."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test empty response chunk without thinking
        empty_response_chunk = {
            "model": "qwen3:4b",
            "created_at": "2025-10-16T11:27:14.82881Z",
            "response": "",
            "done": False,
        }

        result = iterator.chunk_parser(empty_response_chunk)

        # Updated to handle ModelResponseStream return type
        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert result.choices[0].delta.content == None
        assert getattr(result.choices[0].delta, "reasoning_content", None) == ""

    def test_chunk_parser_done_chunk(self):
        """Test that done chunks work correctly."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test done chunk
        done_chunk = {
            "model": "llama2",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        result = iterator.chunk_parser(done_chunk)

        assert result["text"] == ""
        assert result["is_finished"] is True
        assert result["finish_reason"] == "stop"
        assert result["usage"] is not None
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15


class TestOllamaTextCompletionStreamingToolCalls:
    """Regression tests for https://github.com/BerriAI/litellm/issues/35711"""

    def _stream(self, responses, function_call_prompted=True):
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, function_call_prompted=function_call_prompted
        )
        chunks = [
            iterator.chunk_parser({"model": "qwen3", "created_at": "t", "done": False, "response": r})
            for r in responses
        ]
        done = iterator.chunk_parser(
            {
                "model": "qwen3",
                "created_at": "t",
                "done": True,
                "done_reason": "stop",
                "response": "",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        )
        return chunks, done

    def test_streamed_function_call_json_reconstructed_as_tool_call(self):
        chunks, done = self._stream(['{"name": "get_weather",', ' "arguments": {"location": "Paris"}}'])

        for chunk in chunks:
            assert isinstance(chunk, ModelResponseStream)
            assert not chunk.choices[0].delta.content

        assert isinstance(done, ModelResponseStream)
        tool_calls = done.choices[0].delta.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"location": "Paris"}
        assert done.choices[0].finish_reason == "tool_calls"

    def test_leading_whitespace_tokens_do_not_disable_tool_call_reconstruction(self):
        chunks, done = self._stream(["\n", " ", '{"name": "get_weather",', ' "arguments": {"location": "Paris"}}'])

        for chunk in chunks:
            assert isinstance(chunk, ModelResponseStream)
            assert not chunk.choices[0].delta.content

        assert isinstance(done, ModelResponseStream)
        tool_calls = done.choices[0].delta.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"location": "Paris"}
        assert done.choices[0].finish_reason == "tool_calls"

    def test_leading_whitespace_before_plain_text_is_flushed_as_content(self):
        chunks, done = self._stream(["\n", "Hello", " world"])

        streamed = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert streamed == "\nHello world"
        assert done["finish_reason"] == "stop"

    def test_streamed_regular_json_flushed_as_content_once_not_a_function_call(self):
        chunks, done = self._stream(['{"answer":', ' 42}'])

        assert isinstance(chunks[0], ModelResponseStream)
        assert chunks[0].choices[0].delta.content == '{"answer":'
        assert chunks[1].choices[0].delta.content == " 42}"
        assert done["finish_reason"] == "stop"

    def test_streamed_function_call_with_string_arguments_not_double_encoded(self):
        chunks, done = self._stream(['{"name": "get_weather",', ' "arguments": "{\\"location\\": \\"Paris\\"}"}'])

        assert isinstance(done, ModelResponseStream)
        tool_calls = done.choices[0].delta.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert json.loads(tool_calls[0].function.arguments) == {"location": "Paris"}

    def test_brace_prefixed_prose_streams_incrementally(self):
        chunks, done = self._stream(["{note: this", " is not JSON}", " and more text"])

        assert isinstance(chunks[0], ModelResponseStream)
        assert chunks[0].choices[0].delta.content == "{note: this"
        assert chunks[1].choices[0].delta.content == " is not JSON}"
        assert chunks[2].choices[0].delta.content == " and more text"
        assert done["finish_reason"] == "stop"

    def test_plain_text_still_streams_incrementally(self):
        chunks, done = self._stream(["Hello", " world"])

        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
        assert done["finish_reason"] == "stop"

    @pytest.mark.parametrize(
        "arguments_fragment",
        [' "arguments": {"location": "Paris"}}', ' "arguments": "{\\"location\\": \\"Paris\\"}"}'],
    )
    def test_no_tool_call_reconstruction_when_no_tools_were_sent(self, arguments_fragment):
        """A caller that sent no tools must never get a synthesized tool call, and must never lose the
        content it did ask for, even when it requested JSON output."""
        chunks, done = self._stream(['{"name": "get_weather",', arguments_fragment], function_call_prompted=False)

        streamed = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert streamed == '{"name": "get_weather",' + arguments_fragment
        for chunk in chunks:
            assert chunk.choices[0].delta.tool_calls is None
        assert done["finish_reason"] == "stop"


class TestOllamaStreamGating:
    """`utils.py` sets format=json and injects `function_call_prompt` into the messages whenever tools are
    passed to `ollama/`. Only those requests may have their streamed JSON reconstructed into a tool call;
    the iterator learns about it from the request transform."""

    _tool = {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}

    def _iterator_for(self, optional_params, messages=None, sync_stream=True):
        config = OllamaConfig()
        config.transform_request(
            model="qwen3",
            messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        return config.get_model_response_iterator(streaming_response=iter([]), sync_stream=sync_stream)

    def test_tools_request_buffers_a_possible_function_call(self):
        messages = function_call_prompt([{"role": "user", "content": "weather in Paris?"}], [self._tool])

        iterator = self._iterator_for({"format": "json"}, messages=messages)

        assert iterator.function_call_buffering_enabled is True

    def test_tools_request_with_existing_system_message_buffers(self):
        messages = function_call_prompt(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"}], [self._tool]
        )

        iterator = self._iterator_for({"format": "json"}, messages=messages)

        assert iterator.function_call_buffering_enabled is True

    def test_json_output_without_tools_never_buffers(self):
        """response_format=json_object alone must not turn `{"name": ..., "arguments": ...}` content into
        a tool call."""
        iterator = self._iterator_for({"format": "json"})

        assert iterator.function_call_buffering_enabled is False

    def test_plain_request_never_buffers(self):
        iterator = self._iterator_for({"temperature": 0.5})

        assert iterator.function_call_buffering_enabled is False

    @pytest.mark.parametrize("sync_stream", [True, False])
    def test_gate_survives_both_sync_and_async_streaming(self, sync_stream):
        """The async handler builds the iterator without forwarding json_mode, so the flag has to ride
        on the config rather than on that argument."""
        messages = function_call_prompt([{"role": "user", "content": "hi"}], [self._tool])

        iterator = self._iterator_for({"format": "json"}, messages=messages, sync_stream=sync_stream)

        assert iterator.function_call_buffering_enabled is True

    def test_json_output_without_tools_streams_name_arguments_object_as_content(self):
        """End to end through the config: a tool-free JSON request whose schema happens to use top-level
        `name` and `arguments` keys keeps its content and gets no tool call."""
        iterator = self._iterator_for({"format": "json"})
        fragments = ['{"name": "Alice",', ' "arguments": ["x"]}']

        chunks = [
            iterator.chunk_parser({"model": "qwen3", "created_at": "t", "done": False, "response": r})
            for r in fragments
        ]
        done = iterator.chunk_parser(
            {"model": "qwen3", "created_at": "t", "done": True, "done_reason": "stop", "response": ""}
        )

        assert "".join(c.choices[0].delta.content or "" for c in chunks) == "".join(fragments)
        assert all(c.choices[0].delta.tool_calls is None for c in chunks)
        assert done["finish_reason"] == "stop"
