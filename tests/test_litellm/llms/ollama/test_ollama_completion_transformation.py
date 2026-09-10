import json
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm._uuid import uuid
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
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
        assert result.choices[0].delta.content is None
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


async def test_ollama_async_completion_inlines_remote_images_off_the_event_loop(async_only_image_fetch):
    image_url = f"https://img.example/{uuid.uuid4()}.png"
    captured = {}

    def handle(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llava",
                "response": "Green",
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model="ollama/llava",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        api_base="http://ollama.example:11434",
        client=client,
    )

    assert response.choices[0].message.content == "Green"
    assert async_only_image_fetch.fetched == [image_url]
    assert captured["body"]["images"] == [async_only_image_fetch.base64_png]


GRAPH_STATS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "graph_stats",
            "description": "Return node and edge counts of the code graph",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.mark.parametrize(
    "api_base",
    [
        "http://ollama.example:11434",
        "http://ollama.example:11434/",
        "http://ollama.example:11434/api/generate",
        "http://ollama.example:11434/api/generate/",
        "http://ollama.example:11434/api/chat",
    ],
)
def test_ollama_tool_result_turn_is_sent_to_native_chat_api(api_base: str):
    """https://github.com/BerriAI/litellm/issues/40575"""
    requests = []

    def handle(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "message": {"role": "assistant", "content": "The graph has 190921 nodes."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    response = litellm.completion(
        model="ollama/qwen3.8:27b",
        messages=[
            {"role": "user", "content": "How many nodes does the graph have?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "graph_stats", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "graph_stats", "content": '{"nodes": 190921}'},
        ],
        tools=GRAPH_STATS_TOOLS,
        api_base=api_base,
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle))),
    )

    assert [path for path, _ in requests] == ["/api/chat"]
    body = requests[0][1]
    assert body["tools"] == GRAPH_STATS_TOOLS
    assert "format" not in body
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "tool"]
    assert body["messages"][2]["content"] == '{"nodes": 190921}'
    assert response.choices[0].message.content == "The graph has 190921 nodes."
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].finish_reason == "stop"


def test_ollama_streamed_tool_call_is_returned_as_tool_call():
    """https://github.com/BerriAI/litellm/issues/35711"""
    chunks = [
        {
            "model": "qwen3.8:27b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "graph_stats", "arguments": {}}}],
            },
            "done": False,
        },
        {
            "model": "qwen3.8:27b",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    ]

    def handle(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, content="\n".join(json.dumps(chunk) for chunk in chunks).encode())

    streamed = list(
        litellm.completion(
            model="ollama/qwen3.8:27b",
            messages=[{"role": "user", "content": "How many nodes does the graph have?"}],
            tools=GRAPH_STATS_TOOLS,
            stream=True,
            api_base="http://ollama.example:11434",
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle))),
        )
    )

    tool_calls = [tool_call for chunk in streamed for tool_call in chunk.choices[0].delta.tool_calls or []]
    assert [tool_call.function.name for tool_call in tool_calls] == ["graph_stats"]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in streamed) == ""
    assert streamed[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize("empty_parameter", ["none", "tools", "functions"])
def test_ollama_empty_tools_preserve_generate_request(empty_parameter: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert request.url.path == "/api/generate"
        assert "format" not in body
        assert "tools" not in body
        return httpx.Response(200, json={"response": "Hello", "done": True})

    response: Final = litellm.completion(
        model="ollama/qwen3.8:27b",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[] if empty_parameter == "tools" else None,
        functions=[] if empty_parameter == "functions" else None,
        api_base="http://ollama.example:11434/api/generate",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle))),
    )
    assert response.choices[0].message.content == "Hello"


def test_ollama_native_tool_support_error_is_preserved() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(400, json={"error": "model does not support tools"})

    with pytest.raises(litellm.BadRequestError, match="does not support tools"):
        litellm.completion(
            model="ollama/qwen3.8:27b",
            messages=[{"role": "user", "content": "Hello"}],
            tools=GRAPH_STATS_TOOLS,
            api_base="http://ollama.example:11434/api/generate",
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle))),
            num_retries=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_functions", [False, True])
async def test_ollama_async_native_tools(legacy_functions: bool) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert request.url.path == "/prefix/api/chat"
        assert body["tools"] == GRAPH_STATS_TOOLS
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8:27b",
                "message": {"role": "assistant", "content": "Hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        handler: Final = AsyncHTTPHandler()
        await handler.client.aclose()
        handler.client = client
        response: Final = await litellm.acompletion(
            model="ollama/qwen3.8:27b",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None if legacy_functions else GRAPH_STATS_TOOLS,
            functions=[GRAPH_STATS_TOOLS[0]["function"]] if legacy_functions else None,
            api_base="http://ollama.example:11434/prefix/api/generate/",
            client=handler,
        )
    assert response.choices[0].message.content == "Hello"


def test_ollama_add_function_to_prompt_keeps_legacy_json_emulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "add_function_to_prompt", True)
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, json={"response": '{"name": "graph_stats", "arguments": {}}', "done": True, "prompt_eval_count": 1}
        )

    messages: Final = [
        {"role": "system", "content": "You are a graph assistant."},
        {"role": "user", "content": "How many nodes does the graph have?"},
    ]

    response: Final = litellm.completion(
        model="ollama/qwen3.8:27b",
        messages=messages,
        tools=GRAPH_STATS_TOOLS,
        tool_choice="auto",
        api_base="http://ollama.example:11434",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle))),
    )

    assert [path for path, _ in requests] == ["/api/generate"]
    body: Final = requests[0][1]
    assert body["format"] == "json"
    assert "Produce JSON OUTPUT ONLY" in body["prompt"]
    assert "graph_stats" in body["prompt"]
    assert "prompted_functions" not in body["options"]
    assert response.choices[0].message.tool_calls[0].function.name == "graph_stats"
    assert response.choices[0].finish_reason == "tool_calls"
