import json
import os
import sys
from litellm._uuid import uuid
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

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
        assert getattr(result.choices[0].delta, "reasoning_content", None) is ""

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

    def test_chunk_parser_streams_tool_call_instead_of_raw_json(self):
        """A format=json tool call must arrive as tool_calls, not as JSON in the content."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
            tool_names=frozenset({"lookup_account", "lookup"}),
        )

        fragments = [
            '{"',
            "name",
            '": "',
            "lookup_account",
            '", "',
            "arguments",
            '": {"',
            "email",
            '": "',
            "maya.iyer@example.com",
            '"}}',
        ]
        streamed_content = ""
        for fragment in fragments:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            streamed_content += result.choices[0].delta.content or ""

        assert streamed_content == ""

        result = iterator.chunk_parser(
            {
                "model": "qwen2.5:3b",
                "response": "",
                "done": True,
                "prompt_eval_count": 120,
                "eval_count": 24,
            }
        )

        assert result["text"] == ""
        assert result["finish_reason"] == "tool_calls"
        assert result["is_finished"] is True
        assert result["tool_use"] is not None
        assert result["tool_use"]["type"] == "function"
        assert result["tool_use"]["index"] == 0
        assert result["tool_use"]["id"]
        assert result["tool_use"]["function"]["name"] == "lookup_account"
        assert json.loads(result["tool_use"]["function"]["arguments"]) == {"email": "maya.iyer@example.com"}

    def test_chunk_parser_streams_prose_incrementally(self):
        """Prose must keep streaming fragment by fragment and carry no tool call."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
            tool_names=frozenset({"lookup_account", "lookup"}),
        )

        fragments = ["I ", "will ", "look ", "that ", "up", "."]
        deltas = []
        for fragment in fragments:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            deltas.append(result.choices[0].delta.content)

        assert deltas == fragments

        result = iterator.chunk_parser(
            {
                "model": "qwen2.5:3b",
                "response": "",
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 6,
            }
        )

        assert result["text"] == ""
        assert result["finish_reason"] == "stop"
        assert result.get("tool_use") is None

    def test_chunk_parser_releases_non_tool_call_json_as_content(self):
        """A JSON body that is not a tool call must still be delivered as content."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
            tool_names=frozenset({"lookup_account", "lookup"}),
        )

        fragments = ['{"', "city", '": "', "Chennai", '"}']
        streamed_content = ""
        for fragment in fragments:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            streamed_content += result.choices[0].delta.content or ""

        result = iterator.chunk_parser(
            {
                "model": "qwen2.5:3b",
                "response": "",
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 6,
            }
        )

        assert streamed_content + result["text"] == '{"city": "Chennai"}'
        assert result["finish_reason"] == "stop"
        assert result.get("tool_use") is None

    def test_chunk_parser_leaves_json_carrying_extra_keys_as_content(self):
        """A JSON body that merely happens to carry a name must not be mistaken for a tool call."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
            tool_names=frozenset({"lookup_account", "lookup"}),
        )

        body = '{"name": "Chennai", "arguments": {"x": 1}, "population": 7000000}'
        streamed_content = ""
        for fragment in [body[i : i + 6] for i in range(0, len(body), 6)]:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            streamed_content += result.choices[0].delta.content or ""

        result = iterator.chunk_parser(
            {"model": "qwen2.5:3b", "response": "", "done": True, "prompt_eval_count": 5, "eval_count": 9}
        )

        assert streamed_content + result["text"] == body
        assert result["finish_reason"] == "stop"
        assert result.get("tool_use") is None

    def test_chunk_parser_leaves_non_object_arguments_as_content(self):
        """`arguments` must be the object the tool prompt asks for, not any value that happens to be there."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
            tool_names=frozenset({"lookup_account", "lookup"}),
        )

        body = '{"name": "lookup", "arguments": "not an object"}'
        streamed_content = ""
        for fragment in [body[i : i + 6] for i in range(0, len(body), 6)]:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            streamed_content += result.choices[0].delta.content or ""

        result = iterator.chunk_parser(
            {"model": "qwen2.5:3b", "response": "", "done": True, "prompt_eval_count": 5, "eval_count": 9}
        )

        assert streamed_content + result["text"] == body
        assert result["finish_reason"] == "stop"
        assert result.get("tool_use") is None


TOOL_CALL_BODY = '{"name": "lookup_account", "arguments": {"email": "maya.iyer@example.com"}}'

LOOKUP_ACCOUNT_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_account",
        "description": "Look up a customer account by email address",
        "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
    },
}


def _generate_transport():
    """Stands in for ollama's /api/generate, fragmenting the body the way it really does."""
    import httpx

    done = {"model": "qwen2.5:3b", "response": "", "done": True, "prompt_eval_count": 146, "eval_count": 21}

    def handler(request: "httpx.Request") -> "httpx.Response":
        fragments = [TOOL_CALL_BODY[i : i + 4] for i in range(0, len(TOOL_CALL_BODY), 4)]
        lines = [json.dumps({"model": "qwen2.5:3b", "response": f, "done": False}) for f in fragments]
        lines.append(json.dumps(done))
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    return httpx.MockTransport(handler)


class TestOllamaStreamingToolCallsEndToEnd:
    def test_streamed_tool_call_is_not_delivered_as_content(self):
        """The reported bug: a streamed tool call reaching the caller as raw JSON in the content."""
        import httpx

        from litellm import completion
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler(client=httpx.Client(transport=_generate_transport()))

        response = completion(
            model="ollama/qwen2.5:3b",
            messages=[{"role": "user", "content": "Cancel the subscription for maya.iyer@example.com"}],
            tools=[LOOKUP_ACCOUNT_TOOL],
            api_base="http://localhost:11434",
            stream=True,
            client=client,
        )

        streamed_content = ""
        tool_calls = []
        finish_reason = None
        for chunk in response:
            delta = chunk.choices[0].delta
            streamed_content += getattr(delta, "content", None) or ""
            tool_calls += getattr(delta, "tool_calls", None) or []
            finish_reason = chunk.choices[0].finish_reason or finish_reason

        assert streamed_content == ""
        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "lookup_account"
        assert json.loads(tool_calls[0].function.arguments) == {"email": "maya.iyer@example.com"}

    def test_detection_is_off_for_a_request_that_offered_no_tools(self):
        """No tools were rewritten into a prompt, so a tool-shaped body is just content."""
        iterator = OllamaTextCompletionResponseIterator(streaming_response=iter([]), sync_stream=True, json_mode=False)

        streamed_content = ""
        for fragment in [TOOL_CALL_BODY[i : i + 6] for i in range(0, len(TOOL_CALL_BODY), 6)]:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            streamed_content += result.choices[0].delta.content or ""

        result = iterator.chunk_parser(
            {"model": "qwen2.5:3b", "response": "", "done": True, "prompt_eval_count": 5, "eval_count": 9}
        )

        assert streamed_content == TOOL_CALL_BODY
        assert result["finish_reason"] == "stop"
        assert result.get("tool_use") is None

    def test_detection_is_armed_only_with_the_functions_the_request_offered(self):
        """`get_optional_params` passes on the functions it rewrote; those are the only ones accepted."""
        from litellm.utils import get_optional_params

        config = OllamaConfig()
        base = {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_params": {},
            "headers": {},
        }

        json_mode_only = get_optional_params(
            model="qwen2.5:3b",
            custom_llm_provider="ollama",
            stream=True,
            response_format={"type": "json_object"},
        )
        assert "prompted_tool_calls" not in json_mode_only
        config.transform_request(optional_params=json_mode_only, **base)
        assert config.get_model_response_iterator(iter([]), sync_stream=True).tool_names == frozenset()

        with_tools = get_optional_params(
            model="qwen2.5:3b",
            custom_llm_provider="ollama",
            stream=True,
            tools=[LOOKUP_ACCOUNT_TOOL],
        )
        assert with_tools["prompted_tool_calls"] == [LOOKUP_ACCOUNT_TOOL]
        config.transform_request(optional_params=with_tools, **base)
        assert config.get_model_response_iterator(iter([]), sync_stream=True).tool_names == frozenset(
            {"lookup_account"}
        )

    def test_a_function_the_request_never_offered_stays_content(self):
        """A body naming some other function must not be synthesised into a tool call."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False, tool_names=frozenset({"lookup_account"})
        )

        body = '{"name": "delete_account", "arguments": {"email": "maya.iyer@example.com"}}'
        streamed_content = ""
        for fragment in [body[i : i + 6] for i in range(0, len(body), 6)]:
            result = iterator.chunk_parser({"model": "qwen2.5:3b", "response": fragment, "done": False})
            assert isinstance(result, ModelResponseStream)
            assert result.choices and result.choices[0].delta is not None
            streamed_content += result.choices[0].delta.content or ""

        result = iterator.chunk_parser(
            {"model": "qwen2.5:3b", "response": "", "done": True, "prompt_eval_count": 5, "eval_count": 9}
        )

        assert streamed_content + result["text"] == body
        assert result["finish_reason"] == "stop"
        assert result.get("tool_use") is None

    def test_prompted_tool_calls_marker_is_not_sent_to_ollama(self):
        """The marker is litellm-internal; it must not leak into the request body."""
        config = OllamaConfig()

        data = config.transform_request(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"stream": True, "format": "json", "prompted_tool_calls": True},
            litellm_params={},
            headers={},
        )

        assert "prompted_tool_calls" not in data
        assert "prompted_tool_calls" not in data["options"]
